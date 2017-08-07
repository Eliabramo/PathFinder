import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Qnetwork():
    def __init__(self, state_size, h_size, rnn_cell, myScope):
        with tf.name_scope(myScope):
            # The network recieves a frame from the game, flattened into an array.
            # It then resizes it and processes it through four convolutional layers.
            self.trainLength = tf.placeholder(dtype=tf.int32)
            self.batch_size = tf.placeholder(dtype=tf.int32)

            #self.scalarInput = tf.placeholder(shape=[self.batch_size*self.trainLength, state_size], dtype=tf.float32)
            self.scalarInput = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            with tf.name_scope('layer0'):
                self.W0 = tf.Variable(tf.random_normal([state_size, h_size]), name='W0')
                self.B0 = tf.Variable(tf.random_normal([self.batch_size*self.trainLength, h_size]), name='B0')
                self.L0 = tf.nn.relu(tf.matmul(self.scalarInput, self.W0) + self.B0)

            # We take the input and send it to a recurrent layer.
            # The input must be reshaped into [batch x trace x units] for rnn processing,
            # and then returned to [batch x units] when sent through the upper levels.

            self.rnn_input = tf.reshape(self.L0, [self.batch_size, self.trainLength, h_size])
            self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_input, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
            self.rnn = tf.reshape(self.rnn, shape=[-1, state_size])
            # The output from the recurrent player is then split into separate Value and Advantage streams
            self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
            self.AW = tf.Variable(tf.random_normal([state_size // 2, 2]))
            self.VW = tf.Variable(tf.random_normal([state_size // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

            self.salience = tf.gradients(self.Advantage, self.scalarInput)
            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
            self.predict = tf.argmax(self.Qout, 1)

            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, 2, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

            self.td_error = tf.square(self.targetQ - self.Q)

            # In order to only propogate accurate gradients through the network, we will mask the first
            # half of the losses for each trace as per Lample & Chatlot 2016
            self.maskA = tf.zeros([self.batch_size, self.trainLength // 2])
            self.maskB = tf.ones([self.batch_size, self.trainLength // 2])
            self.mask = tf.concat([self.maskA, self.maskB], 1)
            self.mask = tf.reshape(self.mask, [-1])
            self.loss = tf.reduce_mean(self.td_error * self.mask)

            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)
