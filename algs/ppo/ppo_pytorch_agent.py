import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from utils.memory import PPOMemory

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, layers, chkpt_dir='.\\tmp\\ppo'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], n_actions),
            nn.Softmax(dim=-1)
        )
        # f = 0.1
        # T.nn.init.orthogonal_(self.actor.weight.data, f)
        # T.nn.init.orthogonal_(self.actor.bias.data, f)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, layers, chkpt_dir='.\\tmp\\ppo'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], 1)
        )
        # f = 0.1
        # T.nn.init.orthogonal_(self.critic.weight.data, f)
        # T.nn.init.orthogonal_(self.critic.bias.data, f)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value


class Agent:
    def __init__(self, input_dims, n_actions, layers,
                 gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=4):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(n_actions, input_dims, alpha, layers)
        self.critic = CriticNetwork(input_dims, alpha, layers)
        self.memory = PPOMemory(batch_size)

    def store(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        # self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        # self.actor.train()
        return action, probs, value

    def learn(self, values_hist=None, adv_hist=None):
        state_arr, action_arr, old_probs_arr, vals_arr, \
        reward_arr, done_arr, batches = self.memory.generate_batches()
        if len(state_arr) < 2:
            return
        values = vals_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                   (1 - int(done_arr[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = T.tensor(advantage).to(self.actor.device)
        # print(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        values = T.tensor(values).to(self.actor.device)
        # advantage = advantage - values
        if values_hist is not None:
            values_hist.append(values.mean())
        if adv_hist is not None:
            adv_hist.append(advantage.mean())
        for _ in range(self.n_epochs):
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                # prob_ratio = new_probs.exp() / old_probs.exp()
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                entropy = -dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()






