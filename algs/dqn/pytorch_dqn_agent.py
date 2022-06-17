from utils.memory import Memory
from algs.agent import Agent
import numpy.random as rnd
from models import pytorch_models
import torch
import torch.nn.functional as F

class PytorchDqnAgent(Agent):
    def __init__(self, state_space, action_space, layers, gamma=0.99, tau=0.95, memory_size=20000, batch_size=32):
        super().__init__(state_space, action_space)
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.95
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.memory = Memory(memory_size)
        self.layers = layers

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        # self.device = torch.device('cpu')
        self.policy_network = pytorch_models.build_dense(self.state_space, self.action_space, self.layers, policy=True)
        self.target_network = pytorch_models.build_dense(self.state_space, self.action_space, self.layers)
        self.policy_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.eval()
        # self.optimizer = torch.optim.Adam(self.policy_network.parameters())

    def act(self, state):
        return self.epsi_policy(state)

    def epsi_policy(self, state):
        if rnd.random() <= self.epsilon:
            act = rnd.randint(self.action_space)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            act_values = self.policy_network(state_tensor)
            act = torch.argmax(act_values).item()
        return act

    def replay(self):
        if self.memory.size() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device).view(self.batch_size,1)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        predict_y = self.policy_network(states).gather(1, actions)
        target_y = self.target_network(next_states).max(1)[0].detach()
        target_y = rewards + self.gamma * target_y * (1 - dones)
        loss = self.policy_network.loss(predict_y, target_y.unsqueeze(1))
        # policy_weight = self.policy_network.state_dict().values()
        # target_weight = self.target_network.state_dict().values()

        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        # new_policy_weight = self.policy_network.state_dict()
        # new_target_weight = self.target_network.state_dict()
        # self.update_target()



    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def update_target(self):
        weights = self.policy_network.state_dict()
        target_weights = self.target_network.state_dict()
        for k in weights.keys():
            target_weights[k] = (1-self.tau) * target_weights[k] + self.tau * (weights[k])
        self.target_network.load_state_dict(weights)


    def end_episode(self):
        self.update_target()
        self.decay()