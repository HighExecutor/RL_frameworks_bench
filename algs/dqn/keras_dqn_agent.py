import numpy as np
from utils.memory import Memory
from algs.agent import Agent
from models import keras_models
import numpy.random as rnd


class KerasDqnAgent(Agent):
    def __init__(self, state_space, action_space, layers, gamma=0.99, tau=0.95, memory_size=20000, batch_size=32):
        super().__init__(state_space, action_space)
        self.policy_func = self.epsi_policy
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.95
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.memory = Memory(memory_size)
        self.layers = layers
        self.policy_network = keras_models.build_dense(self.state_space, self.action_space, self.layers)
        self.target_network = keras_models.build_dense(self.state_space, self.action_space, self.layers)
        self.update_target()

    def act(self, state):
        return self.policy_func(state)

    def epsi_policy(self, state):
        if rnd.random() <= self.epsilon:
            act = rnd.randint(self.action_space)
        else:
            act_values = self.policy_network.predict_on_batch(state.reshape(1, self.state_space))[0]
            act = np.argmax(act_values)
        return act

    def bolts_policy(self, state):
        q_values = self.policy_network.predict_on_batch(state)
        exp_values = np.exp(q_values)
        probs = exp_values / np.sum(exp_values)
        action = rnd.choice(range(self.action_space), p=probs)
        return action

    def replay(self):
        if self.memory.size() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        indices = np.arange(self.batch_size)
        target_y = self.target_network.predict_on_batch(next_states)
        target_y = np.amax(target_y, axis=1)
        target_y = rewards + (1 - dones) * self.gamma * target_y
        predict_y = self.policy_network.predict_on_batch(states)
        predict_y[indices, actions] = target_y
        self.policy_network.train_on_batch(states, predict_y)

    def end_episode(self):
        self.update_target()
        self.decay()

    def update_target(self):
        policy_weights = np.array(self.policy_network.get_weights()).copy()
        target_weights = (1 - self.tau) * np.array(self.target_network.get_weights()) + self.tau * policy_weights
        self.target_network.set_weights(target_weights)

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
