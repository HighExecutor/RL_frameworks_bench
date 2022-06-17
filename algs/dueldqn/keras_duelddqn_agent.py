import numpy as np
from utils.memory import Memory
from models import keras_models
import numpy.random as rnd


class KerasDuelDdqnAgent:
    def __init__(self, state_space, action_space, layers, gamma=0.99, tau=0.95, memory_size=20000, batch_size=32):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.95
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.memory = Memory(memory_size)
        self.layers = layers
        self.policy_network = keras_models.build_dual(self.state_space, self.action_space, self.layers)
        self.target_network = keras_models.build_dual(self.state_space, self.action_space, self.layers)
        self.update_target()

    def act(self, state):
        if rnd.random() <= self.epsilon:
            act = rnd.randint(self.action_space)
            act_values = None
        else:
            act_values = self.policy_network.predict_on_batch(state.reshape(1, self.state_space))[0]
            act = np.argmax(act_values)
        return act, act_values

    def replay(self):
        if self.memory.size() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        indices = np.arange(self.batch_size)

        t_next = self.target_network.predict_on_batch(next_states)
        q_next = self.policy_network.predict_on_batch(next_states)
        q_pred = self.policy_network.predict_on_batch(states)

        q_next_actions = np.argmax(q_next, axis=1)
        target_y = t_next[indices, q_next_actions]
        target_y = rewards + (1 - dones) * self.gamma * target_y
        q_pred[indices, actions] = target_y
        self.policy_network.train_on_batch(states, q_pred)

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

    def remember(self, state, action, reward, next_state, dones):
        self.memory.remember(state, action, reward, next_state, dones)
