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
import agent
from helper import *

class experience_buffer():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])

# the game environment
num_enemies = 0
enemies_speed = 3
env = running_balls_env.running_balls_env(num_enemies, enemies_speed)

# Setting the training parameters
batch_size = 4  # How many experience traces to use for each training step.
trace_length = 8  # How long each experience trace will be when training
update_freq = 5  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 1.0  # Starting chance of random action
endE = 0.1  # Final chance of random action
num_episodes = 1000  # How many episodes of game environment to train network with.
max_epLength = 1000  # The max allowed length of our episode.
anneling_steps = max_epLength*500  # How many steps of training to reduce startE to endE.
pre_train_steps = batch_size*max_epLength  # How many steps of random actions before training begins.
load_model = False #True  # Whether to load a saved model.
log_path = "./train"  # The path to save our model to.
state_size = env.state_size  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
action_size = env.action_size
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
time_per_step = 1  # Length of each step used in gif creation
summaryPeriod = num_episodes/100        # number of episodes before save summary
checkpointPeriod = num_episodes/10      # number of episodes before save check point
learning_rate = 0.01
tau = 0.001

tf.reset_default_graph()
# We define the cells for the primary and target q-networks
cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainQN = agent.Qnetwork(state_size, h_size, cell, learning_rate, 'main')
targetQN = agent.Qnetwork(state_size, h_size, cellT, learning_rate, 'target')

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / anneling_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(log_path):
    os.makedirs(log_path)

j_ = tf.placeholder(tf.float32)
d_ = tf.placeholder(tf.float32)
rAll_ = tf.placeholder(tf.float32)
e_ = tf.placeholder(tf.float32)
total_steps_ = tf.placeholder(tf.float32)
total_actions_ = tf.placeholder(tf.float32)
tf.summary.scalar('episode_length', j_)
tf.summary.scalar('winnings', d_)
tf.summary.scalar('total_reward', rAll_)
tf.summary.scalar('e_greedy', e_)
tf.summary.scalar('total_steps', total_steps_)
tf.summary.histogram('total_actions', total_actions_)
merged = tf.summary.merge_all()

try:
    sess = tf.Session()
except:
    pass

with tf.Session() as sess:
    if load_model == True and os.path.exists(log_path + '*.ckpt'):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(log_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(init)

    writer = tf.summary.FileWriter(log_path, sess.graph)

    updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
    for i in range(num_episodes):
        print('i=%d' % i)
        episodeBuffer = []
        # Reset environment and get first new observation
        s = env.reset()
        d = 0
        rAll = 0
        j = 0
        aList = np.zeros([1, max_epLength])
        state = (np.zeros([1, h_size]), np.zeros([1, h_size]))  # Reset the recurrent layer's hidden state
        # The Q-Network
        while j < max_epLength:
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                state1 = sess.run(mainQN.rnn_state, feed_dict={mainQN.scalarInput: s, mainQN.trainLength: 1, mainQN.state_in: state, mainQN.batch_size: 1})
                a = np.random.randint(0, action_size)
            else:
                a, state1 = sess.run([mainQN.predict, mainQN.rnn_state], feed_dict={mainQN.scalarInput: s, mainQN.trainLength: 1, mainQN.state_in: state, mainQN.batch_size: 1})
                a = a[0]
            s1, r, d = env.step(a)
            total_steps += 1
            episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    updateTarget(targetOps, sess)
                    # Reset the recurrent layer's hidden state
                    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))

                    trainBatch = myBuffer.sample(batch_size, trace_length)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    # trainBatch[:, 3] is s1's vector
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3]), mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3]), targetQN.trainLength: trace_length, targetQN.state_in: state_train, targetQN.batch_size: batch_size})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size * trace_length), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    # trainBatch[:, 0] is s's vector
                    # trainBatch[:, 1] is a's vector
                    sess.run(mainQN.updateModel, \
                             feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ, \
                                        mainQN.actions: trainBatch[:, 1], mainQN.trainLength: trace_length, \
                                        mainQN.state_in: state_train, mainQN.batch_size: batch_size})

            rAll += r
            s = s1
            state = state1
            aList[0,j] = a
            if d == True:
                break
            #env.render()
            j += 1

        # Add the episode to the experience buffer
        bufferArray = np.array(episodeBuffer)
        episodeBuffer = list(zip(bufferArray))
        if len(episodeBuffer) >= trace_length:      # don't add short episodes
            myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll)

        # Periodically save the model.
        if i % summaryPeriod == 0 and i != 0:
            _, _, _, _, _, _, summary = sess.run([j_, d_, rAll_, e_, total_steps_, total_actions_, merged], feed_dict={j_: j, d_: d, rAll_: rAll, e_: e, total_steps_: total_steps, total_actions_: aList})
            writer.add_summary(summary, i)
        if i % checkpointPeriod == 0 and i != 0:
            saver.save(sess, log_path + '/model-' + str(i) + '.ckpt')
            print("Saved Model")
    saver.save(sess, log_path + '/model-' + str(i) + '.ckpt')