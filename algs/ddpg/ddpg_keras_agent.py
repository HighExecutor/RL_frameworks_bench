import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, InputLayer, Multiply, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from utils.memory import Memory
import tensorflow as tf
import tensorflow.keras.backend as K


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


class Critic(Sequential):
    def __init__(self, input_size, action_size, layers):
        super().__init__()
        # self.state_layer = InputLayer(input_shape=(input_size, ))
        # self.action_layer = InputLayer(input_shape=(action_size, ))
        self.add(InputLayer(input_shape=(input_size + action_size,)))
        self.add(Dense(layers[0], activation='relu'))
        self.add(Dense(layers[1], activation='relu'))
        self.add(Dense(1, activation=None))
        # self.in_l = InputLayer(input_shape=(input_size + action_size, ))
        # self.fc1 = Dense(layers[0], activation='relu')
        # self.fc2 = Dense(layers[1], activation='relu')
        # self.q = Dense(1, activation=None)

    # def call(self, s_a, **kwargs):
    #     # state, action = x
    #     # s = self.state_layer(state)
    #     # a = self.action_layer(action)
    #     # merged = tf.concat([s, a], axis=1)
    #     x = self.in_l(s_a)
    #     a_val = self.fc1(x)
    #     a_val = self.fc2(a_val)
    #     q = self.q(a_val)
    #     return q


class CustomLoss:
    def __init__(self, critic, state_size, action_size):
        self.critic = critic
        self.states_buffer = tf.zeros(dtype=tf.float32, shape=(64, state_size))

    def __call__(self, empty, y_predict, *args, **kwargs):
        policy_actions = y_predict
        actor_loss = - self.critic(tf.concat([self.states_buffer, policy_actions], axis=1))
        return tf.reduce_mean(actor_loss)


class Actor(Sequential):
    def __init__(self, state_size, action_size, layers):
        super().__init__()
        # self.input_layer = InputLayer(input_shape=(state_size,))
        # self.fc1 = Dense(layers[0], activation='relu')
        # self.fc2 = Dense(layers[1], activation='relu')
        # self.mu = Dense(action_size, activation='tanh')
        self.add(InputLayer(input_shape=(state_size,)))
        self.add(Dense(layers[0], activation='relu'))
        self.add(Dense(layers[1], activation='relu'))
        self.add(Dense(action_size, activation='tanh'))

    # def actor_loss(self, empty, y_predict):
    #     policy_actions = y_predict
    #     actor_loss = - self.critic(tf.concat([self.states_buffer, policy_actions], axis=1))
    #     return tf.reduce_mean(actor_loss)

    # def call(self, state, **kwargs):
    #     prob = self.input_layer(state)
    #     prob = self.fc1(prob)
    #     prob = self.fc2(prob)
    #     mu = self.mu(prob)
    #     return mu


class KerasDdpgAgent:
    def __init__(self, state_size, action_size, action_min, action_max, layers, lr1, lr2, gamma, tau, mem_size,
                 batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max
        self.lr1 = lr1
        self.lr2 = lr2
        self.batch_size = batch_size
        self.layers = layers
        self.gamma = gamma
        self.tau = tau
        self.memory = Memory(mem_size)
        self.noise = OUActionNoise(mu=np.zeros(action_size))

        self.critic = Critic(state_size, action_size, layers)
        self.critic.compile(optimizer=Adam(lr=lr2), loss='mse')
        self.target_critic = Critic(state_size, action_size, layers)
        self.target_critic.compile(optimizer=Adam(lr=lr2), loss='mse')

        self.loss = CustomLoss(self.critic, state_size, action_size)

        self.actor = Actor(state_size, action_size, layers)
        self.actor.compile(optimizer=Adam(lr=lr1), loss=self.loss)
        self.target_actor = Actor(state_size, action_size, layers)
        self.target_actor.compile(optimizer=Adam(lr=lr1), loss='mse')

        self.update_networks()

    def remember(self, state, action, reward, new_state, done):
        self.memory.remember(state, action, reward, new_state, done)

    def update_networks(self):
        weights = []
        targets = self.target_actor.get_weights()
        for i, weight in enumerate(self.actor.get_weights()):
            weights.append(weight * self.tau + targets[i] * (1 - self.tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.get_weights()
        for i, weight in enumerate(self.critic.get_weights()):
            weights.append(weight * self.tau + targets[i] * (1 - self.tau))
        self.target_critic.set_weights(weights)

    def act(self, state):
        actions = self.actor.predict_on_batch(state.reshape(1, self.state_size))[0]
        actions += self.noise()
        actions = np.clip(actions, self.action_min, self.action_max)
        return actions

    def learn(self):
        if self.memory.size() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.reshape(self.batch_size, self.state_size)
        actions = actions.reshape(self.batch_size, self.action_size)
        next_states = next_states.reshape(self.batch_size, self.state_size)
        rewards = rewards.reshape(self.batch_size, 1)
        dones = dones.reshape(self.batch_size, 1)

        target_actions = self.target_actor.predict_on_batch(next_states)
        target_critic = self.target_critic.predict_on_batch(np.concatenate([next_states, target_actions], axis=1))
        # critic = self.critic(states, actions)
        target = rewards + self.gamma * target_critic * (1 - dones)
        self.critic.train_on_batch(np.concatenate([states, actions], axis=1), target)

        # policy_actions = self.actor(states)

        # actor_loss = - self.critic([states, policy_actions])
        states_tensor = tf.convert_to_tensor(states)
        self.loss.state_buffer = states_tensor
        predict_tensor = tf.zeros(shape=(self.batch_size, self.action_size))
        self.actor.train_on_batch(states_tensor, predict_tensor)
        self.actor.state_buffer = None

    def end_episode(self):
        self.update_networks()
