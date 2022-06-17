import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class ActorCritic(nn.Module):
    def __init__(self, state_space, action_space, layers, device):
        super(ActorCritic, self).__init__()
        self.critic_linear1 = nn.Linear(state_space, layers[0])
        self.critic_linear2 = nn.Linear(layers[0], layers[1])
        self.critic_linear3 = nn.Linear(layers[1], 1)

        self.actor_linear1 = nn.Linear(state_space, layers[0])
        self.actor_linear2 = nn.Linear(layers[0], layers[1])
        self.actor_linear3 = nn.Linear(layers[1], action_space)
        self.device = device

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim=1)

        return value, policy_dist

class PytorchA2cAgent:
    def __init__(self, state_space, action_space, layers, gamma=0.99):
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model = ActorCritic(state_space, action_space, layers, self.device)
        self.model.optimizer = optim.Adam(self.model.parameters())
        self.model.to(self.device)

        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropy_term = 0

    def act(self, state):
        value, policy_dist = self.model.forward(state)
        value = value.detach().to('cpu').item()
        dist = policy_dist.detach().to('cpu').numpy()
        action = np.random.choice(self.action_space, p=np.squeeze(dist))
        log_prob = torch.log(policy_dist.squeeze(0)[action])

        entropy = -np.sum(np.mean(dist) * np.log(dist))
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropy_term += entropy
        return action, (value, dist)

    def store_rewards(self, reward):
        self.rewards.append(reward)

    def learn(self, last_state):
        qval, _ = self.model.forward(last_state)
        qval = qval.detach().to('cpu').numpy()[0,0]
        qvals = np.zeros_like(self.values)
        for t in reversed(range(len(self.rewards))):
            qval = self.rewards[t] + self.gamma * qval
            qvals[t] = qval

        values = torch.FloatTensor(self.values).to(self.device)
        qvals = torch.FloatTensor(qvals).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)

        advantage = qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001*self.entropy_term
        self.model.optimizer.zero_grad()
        ac_loss.backward()
        self.model.optimizer.step()

    def end_episode(self):
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.entropy_term = 0






