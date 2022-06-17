import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from utils.memory import Memory


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)


class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, action_bound, batch_size=64):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)

        self.unnormalized_actor_gradients = tf.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize = optimizers.Adam(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, self.input_dims],
                                        name='inputs')

            self.action_gradient = tf.placeholder(tf.float32,
                                                  shape=[None, self.n_actions],
                                                  name='gradients')

            f1 = 1. / np.sqrt(self.fc1_dims)
            init1 = RandomUniform(-f1, f1)
            dense1 = layers.Dense(units=self.fc1_dims,
                                     kernel_initializer=init1,
                                     bias_initializer=init1)(self.input)
            batch1 = layers.BatchNormalization()(dense1)
            layer1_activation = activations.relu(batch1)

            # f2 = 1. / np.sqrt(self.fc2_dims)
            f2 = 0.002
            init2 = RandomUniform(-f2, f2)
            dense2 = layers.Dense(units=self.fc2_dims,
                                     kernel_initializer=init2,
                                     bias_initializer=init2)(layer1_activation)
            batch2 = layers.BatchNormalization()(dense2)
            layer2_activation = activations.relu(batch2)

            f3 = 0.004
            init3 = RandomUniform(-f3, f3)
            mu = layers.Dense(units=self.n_actions,
                                 activation='tanh',
                                 kernel_initializer=init3,
                                 bias_initializer=init3)(layer2_activation)
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                 self.action_gradient: gradients})


class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims,
                 batch_size=64):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)

        self.optimize = optimizers.Adam(self.lr).minimize(self.loss, [])

        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, self.input_dims],
                                        name='inputs')

            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='actions')

            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, 1],
                                           name='targets')

            f1 = 1. / np.sqrt(self.fc1_dims)
            dense1 = layers.Dense(units=self.fc1_dims,
                                     kernel_initializer=RandomUniform(-f1, f1),
                                     bias_initializer=RandomUniform(-f1, f1))(self.input)
            batch1 = layers.BatchNormalization()(dense1)
            layer1_activation = activations.relu(batch1)
            # f2 = 1. / np.sqrt(self.fc2_dims)
            f2 = 0.002
            dense2 = layers.Dense(units=self.fc2_dims,
                                     kernel_initializer=RandomUniform(-f2, f2),
                                     bias_initializer=RandomUniform(-f2, f2))(layer1_activation)
            batch2 = layers.BatchNormalization()(dense2)

            action_in = layers.Dense(units=self.fc2_dims,
                                        activation='relu')(self.actions)

            state_actions = tf.add(batch2, action_in)
            state_actions = activations.relu(state_actions)

            f3 = 0.004
            self.q = layers.Dense(units=1,
                                     kernel_initializer=RandomUniform(-f3, f3),
                                     bias_initializer=RandomUniform(-f3, f3))(state_actions)

            # self.loss = tf.losses.mean_squared_error(self.q_target, self.q)
            self.loss = lambda: tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize,
                             feed_dict={self.input: inputs,
                                        self.actions: actions,
                                        self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = Memory(max_size)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess,
                           layer1_size, layer2_size, env.action_space.high)
        self.critic = Critic(beta, n_actions, 'Critic', input_dims, self.sess,
                             layer1_size, layer2_size)

        self.target_actor = Actor(alpha, n_actions, 'TargetActor',
                                  input_dims, self.sess, layer1_size,
                                  layer2_size, env.action_space.high)
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims,
                                    self.sess, layer1_size, layer2_size)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # define ops here in __init__ otherwise time to execute the op
        # increases with each execution.
        self.update_critic = \
        [self.target_critic.params[i].assign(
                      tf.multiply(self.critic.params[i], self.tau) \
                    + tf.multiply(self.target_critic.params[i], 1. - self.tau))
         for i in range(len(self.target_critic.params))]

        self.update_actor = \
        [self.target_actor.params[i].assign(
                      tf.multiply(self.actor.params[i], self.tau) \
                    + tf.multiply(self.target_actor.params[i], 1. - self.tau))
         for i in range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.remember(state, action, reward, new_state, done)

    def act(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state) # returns list of list
        noise = self.noise()
        mu_prime = mu + noise

        return mu_prime[0]

    def learn(self):
        if self.memory.size() < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample(self.batch_size)

        critic_value_ = self.target_critic.predict(new_state,
                                           self.target_actor.predict(new_state))
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*(1-done[j]))
        target = np.reshape(target, (self.batch_size, 1))

        _ = self.critic.train(state, action, target)

        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)

        self.actor.train(state, grads[0])

    def end_episode(self):
        self.update_network_parameters()
