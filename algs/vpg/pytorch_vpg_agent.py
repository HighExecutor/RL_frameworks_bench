import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DenseNetwork(nn.Module):
    def __init__(self, state_space, action_space, layers):
        super(DenseNetwork, self).__init__()
        self.input = nn.Linear(state_space, layers[0])
        self.ls = len(layers)
        if self.ls > 1:
            self.l1 = nn.Linear(layers[0], layers[1])
        if self.ls > 2:
            self.l2 = nn.Linear(layers[1], layers[2])
        self.output = nn.Linear(layers[-1], action_space)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = T.tensor(x).to(self.device)
        x = F.relu(self.input(x))
        if self.ls > 1:
            x = F.relu(self.l1(x))
        if self.ls > 2:
            x = F.relu(self.l2(x))
        x = self.output(x)
        return x

class PytorchVpgAgent:
    def __init__(self, state_space, action_space, layers, gamma=0.99):
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.policy = DenseNetwork(state_space, action_space, layers)

    def act(self, observation):
        probabilities = F.softmax(self.policy.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item(), np.array(action_probs.probs.data.to('cpu'))

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        # Assumes only a single episode for reward_memory
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []