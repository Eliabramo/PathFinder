import running_balls_env
import agent
from helper import *

# ### Testing the network

# the game environment
env = running_balls_env.running_balls_env()

# In[ ]:

e = 0.01  # The chance of chosing a random action
num_episodes = 10000  # How many episodes of game environment to train network with.
load_model = True  # Whether to load a saved model.
train_path = "./train"
test_path = "./test"
# The path to save/load our model to/from.
state_size = env.state_size  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
action_size = env.action_size
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
max_epLength = 1000  # The max allowed length of our episode.
time_per_step = 1  # Length of each step used in gif creation
summaryLength = 100  # Number of epidoes to periodically save for analysis


# In[ ]:

tf.reset_default_graph()
cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainQN = agent.Qnetwork(state_size, h_size, cell, 'main')
targetQN = agent.Qnetwork(state_size, h_size, cellT, 'target')

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(test_path):
    os.makedirs(test_path)

j_ = tf.placeholder(tf.float32)
d_ = tf.placeholder(tf.float32)
rAll_ = tf.placeholder(tf.float32)
tf.summary.scalar('episode_length', j_)
tf.summary.scalar('winnings', d_)
tf.summary.scalar('total_reward', rAll_)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    if load_model == True: #and os.path.exists(train_path + '*.ckpt'):
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(train_path)
        saver.restore(sess ,ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    writer = tf.summary.FileWriter(test_path, sess.graph)

    for i in range(num_episodes):
        print('i=%d' % i)
        # Reset environment and get first new observation
        s = env.reset()
        d = 0
        rAll = 0
        j = 0
        state = (np.zeros([1, h_size]), np.zeros([1, h_size]))  # Reset the recurrent layer's hidden state

        # The Q-Network
        while j < max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial. j+=1
            j += 1
            a, state1 = sess.run([mainQN.predict, mainQN.rnn_state], feed_dict={mainQN.scalarInput:s, mainQN.trainLength:1, mainQN.state_in:state, mainQN.batch_size:1})
            a = a[0]
            s1,r,d = env.step(a)
            total_steps += 1
            rAll += r
            s = s1
            state = state1
            env.render()
            if d == True:
                break


        jList.append(j)
        rList.append(rAll)

        # Periodically save the model.
        if i % 100 == 0 and i != 0:
            _, _, _, summary = sess.run([j_, d_, rAll_, merged], feed_dict={j_: j, d_: d, rAll_: rAll})
            writer.add_summary(summary, i)
        if len(rList) % summaryLength == 0 and len(rList) != 0:
            print (total_steps,np.mean( rList[-summaryLength:]), e)
print ("Percent of succesful episodes: " + str(sum(rList)/ num_episodes) + "%")
