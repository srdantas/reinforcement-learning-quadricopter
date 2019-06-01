import tensorflow as tf
import numpy as np


class Agent:
    def __init__(self, task, session, hp, memory, actor , critic):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.reward_list = []
        self.sess = session

        self.actor = actor
        self.critic = critic

        self.memory = memory
        self.hp = hp
        self.count = 0
        self.total_reward = 0
        self.noise = Noise(self.action_size, 0.0, 0.15, 0.2)
        self.c_loss = 0
        self.a_loss = 0

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        self.state = state
        return state

    def step(self, reward, action, next_state, done):
        self.total_reward += reward
        self.count += 1
        self.memory.add((self.state, action, reward, next_state, done))
        if len(self.memory) > self.hp.batch_size:
            exp = self.memory.sample(self.hp.batch_size)
            self.learn(exp)
        self.state = next_state

    def act(self, state):
        feed = {self.actor.states_: state.reshape((1, *state.shape))}
        action = self.sess.run(self.actor.actions, feed_dict=feed)
        return list(action + self.noise.sample())

    def act_without_noise(self, state):
        feed = {self.actor.states_: state.reshape((1, *state.shape))}
        action = self.sess.run(self.actor.actions, feed_dict=feed)
        return list(action)

    def learn(self, batch):
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch]).astype(
          np.float32).reshape(-1, self.action_size)
        rewards = np.array([each[2] for each in batch]).astype(
          np.float32).reshape(-1, 1)
        next_states = np.array([each[3] for each in batch])
        dones = np.array([each[4] for each in batch]).astype(np.uint8).reshape(-1, 1)

        actions_next = self.sess.run(self.actor.actions,
                                     feed_dict={
                                       self.actor.states_: next_states,
                                       self.actor.dropout: 0
                                     })

        Q_target_next = self.sess.run(self.critic.Q,
                                      feed_dict={
                                        self.critic.states_: next_states,
                                        self.critic.actions_: actions_next,
                                        self.critic.dropout: 0
                                      })
        Q_targets = rewards + self.hp.gamma * Q_target_next * (1-dones)
        c_loss, _, action_gradients =self.sess.run(
          [self.critic.loss, self.critic.opt, self.critic.action_gradients],
          feed_dict={
            self.critic.true:Q_targets,
            self.critic.states_: states,
            self.critic.actions_: actions,
            self.critic.dropout: 0.3,
            self.critic.lr: self.hp.learning_rate
          }
        )
        a_loss, _ = self.sess.run(
          [self.actor.loss, self.actor.opt],
          feed_dict={
            self.actor.states_: states,
            self.actor.action_gradients: action_gradients[0],
            self.actor.dropout: 0.3,
            self.actor.lr: self.hp.learning_rate
          }
        )
        self.c_loss = c_loss
        self.a_loss = a_loss


class Noise:
    def __init__(self, size, x, theta, sigma):
        self.x = x * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.x
    def sample(self):
        state = self.state
        dstate = self.theta * (self.x - state) + self.sigma * np.random.randn(len(state))
        self.state = state + dstate
        return self.state
