from algs.agent import Agent
from utils.memory import PolicyMemory
from models.keras_models import build_pg_network
import numpy as np

class KerasVpgAgent():
    def __init__(self, state_space, action_space, layers, gamma=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.memory = PolicyMemory()
        self.gamma = gamma
        self.G = 0
        self.layers = layers
        self.actions_arr = np.arange(action_space)
        self.policy_net, self.predict_net = build_pg_network(self.state_space, self.action_space, self.layers)

    def remember(self, state, action, reward):
        self.memory.remember(state, action, reward)

    def act(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.predict_net.predict(state)[0]
        action = np.random.choice(self.actions_arr, p=probabilities)

        return action

    def learn(self):
        state_memory = np.array(self.memory.states)
        action_memory = np.array(self.memory.actions)
        reward_memory = np.array(self.memory.rewards)

        mem_size = self.memory.size()
        actions = np.zeros([mem_size, self.action_space])
        actions[np.arange(mem_size), action_memory] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std

        self.policy_net.train_on_batch([state_memory, self.G], actions)


    def end_episode(self):
        self.memory.clear()


