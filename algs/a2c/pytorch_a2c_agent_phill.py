import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_space, action_space, layers, device):
        super(ActorCritic, self).__init__()
        self.layer1 = nn.Linear(state_space, layers[0])
        self.layer2 = nn.Linear(layers[0], layers[1])
        self.critic = nn.Linear(layers[1], 1)
        self.actor = nn.Linear(layers[1], action_space)
        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        pi = self.actor(x)
        v = self.critic(x)
        return pi, v

class PytorchA2cAgent:
    def __init__(self, state_space, action_space, layers, gamma=0.99):
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model = ActorCritic(state_space, action_space, layers, self.device)
        self.log_probs = None


    def act(self, state):
        state = torch.Tensor(state).to(self.device)
        probabilities, value = self.model.forward(state)
        probabilities = F.softmax(probabilities)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item(), (probabilities.to('cpu').data.numpy(), value.to('cpu').item())

    # def store_rewards(self, reward):
    #     self.rewards.append(reward)

    def learn(self, state, reward, next_state, done):
        self.model.optimizer.zero_grad()
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        _, critic_value_ = self.model.forward(next_state)
        _, critic_value = self.model.forward(state)

        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()

        self.model.optimizer.step()

    # def end_episode(self):
        # self.values = []
        # self.rewards = []
        # self.log_probs = None
        # self.entropy_term = 0






