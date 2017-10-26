import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
#import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
import running_balls_env

from helper import *

class Qnetwork():
    def __init__(self, state_size, h_size, rnn_cell, learning_rate, myScope):
        with tf.name_scope(myScope):
            self.trainLength = tf.placeholder(dtype=tf.int32)
            self.batch_size = tf.placeholder(dtype=tf.int32)

            # The network recieves a frame from the game, flattened into an array.
            # It then resizes it and processes it through four convolutional layers.
            self.scalarInput = tf.placeholder(shape=[None, state_size], dtype=tf.float32)

            with tf.name_scope('FC0'):
                self.W0 = tf.Variable(tf.random_normal([state_size, h_size]), name='W0')
                self.B0 = tf.Variable(tf.random_normal([h_size]), name='B0')
                self.FC0out = tf.nn.relu(tf.matmul(self.scalarInput, self.W0) + self.B0, name='FC0out')
                tf.summary.histogram('W0',self.W0)
                tf.summary.histogram('B0', self.B0)

            with tf.name_scope('FC1'):
                self.W1 = tf.Variable(tf.random_normal([h_size, h_size]), name='W1')
                self.B1 = tf.Variable(tf.random_normal([h_size]), name='B1')
                self.FC1out = tf.nn.relu(tf.matmul(self.FC0out, self.W1) + self.B1, name='FC1out')
                tf.summary.histogram('W1', self.W1)
                tf.summary.histogram('B1', self.B1)


            # We take the input and send it to a recurrent layer.
            # The input must be reshaped into [batch x trace x units] for rnn processing,
            # and then returned to [batch x units] when sent through the upper levles.
            with tf.name_scope('RNN'):
                self.rnn_input = tf.reshape(self.FC1out, [self.batch_size, self.trainLength, h_size])
                self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
                self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_input, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
                self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
                #tf.summary.histogram('RNN_state', self.rnn_state)

            # The output from the recurrent player is then split into separate Value and Advantage streams
            with tf.name_scope('Duel'):
                self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
                self.AW = tf.Variable(tf.random_normal([h_size // 2, 2]))
                self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]))
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

            self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.updateModel = self.trainer.minimize(self.loss)
