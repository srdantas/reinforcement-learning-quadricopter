import tensorflow as tf


class Actor:
    def __init__(self, action_range, action_low, state_size=18, action_size=4,
                 name='Actor'):
        with tf.variable_scope(name):
            self.states_ = tf.placeholder(tf.float32,
                                          [None, state_size], name='states')

            self.action_gradients = tf.placeholder(tf.float32,
                                                   [None, action_size],
                                                   name='action_targets')

            self.dropout = tf.placeholder(tf.float32, name='dropout')
            self.lr = tf.placeholder(tf.float32, name='lr')

            self.x1 = tf.layers.dense(self.states_, 32, activation=tf.nn.relu)
            self.x2 = tf.layers.dense(self.x1, 64, activation=tf.nn.relu)
            self.x2 = tf.layers.dense(self.x2, 32, activation=tf.nn.relu)
            self.x2 = tf.layers.dense(self.x2, 64, activation=tf.nn.relu)
            self.x3 = tf.layers.dense(self.x2, 32, activation=tf.nn.relu)
            self.actions_raw = tf.layers.dense(self.x3, action_size,
                                               activation=tf.nn.sigmoid)
            self.actions = tf.add(tf.multiply(self.actions_raw,
                                              action_range),
                                  action_low)

            self.loss = tf.reduce_mean(tf.multiply(-self.action_gradients,
                                                   self.actions))
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


class Critic:
    def __init__(self, state_size=18, action_size=1,
                 name='Critic'):
        with tf.variable_scope(name):
            self.states_ = tf.placeholder(tf.float32,
                                          [None, state_size], name='states')
            self.actions_ = tf.placeholder(tf.float32,
                                           [None, 4], name='actions')

            self.dropout = tf.placeholder(tf.float32, name='dropout')
            self.true = tf.placeholder(tf.float32, name='true')

            reg = tf.contrib.layers.l2_regularizer(scale=0.001)

            self.lr = tf.placeholder(tf.float32, name='lr')

            self.s1 = tf.layers.dense(self.states_,32, activation=tf.nn.relu )
            self.s2 = tf.layers.dense(self.s1, 64, activation=tf.nn.relu)
            self.s2 = tf.layers.batch_normalization(self.s2, training=True)

            self.a1 = tf.layers.dense(self.actions_, 32, activation=tf.nn.relu)
            self.a2 = tf.layers.dense(self.a1, 64, activation=tf.nn.relu)
            self.a2 = tf.layers.batch_normalization(self.a2, training=True)

            self.c = tf.add(self.s2, self.a2)
            self.c1 = tf.layers.dense(self.c, 32, activation=tf.nn.relu)
            self.Q = tf.layers.dense(self.c1, 1,
                                     activation=None,
                                     kernel_regularizer=reg)


            self.action_gradients = tf.gradients(self.Q, self.actions_)

            self.loss = tf.losses.mean_squared_error(self.true, self.Q)
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)