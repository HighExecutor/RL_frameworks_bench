import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class DenseNetwork(nn.Module):
    def __init__(self, state_space, action_space, layers, policy):
        nn.Module.__init__(self)
        self.input = nn.Linear(state_space, layers[0])
        self.ls = len(layers)
        if self.ls > 1:
            self.l1 = nn.Linear(layers[0], layers[1])
        if self.ls > 2:
            self.l2 = nn.Linear(layers[1], layers[2])
        self.output = nn.Linear(layers[-1], action_space)
        if policy:
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
            self.loss = F.mse_loss

    def forward(self, x):
        x = F.relu(self.input(x))
        if self.ls > 1:
            x = F.relu(self.l1(x))
        if self.ls > 2:
            x = F.relu(self.l2(x))
        x = self.output(x)
        return x


def build_dense(state_space, action_space, layers, policy=False):
    return DenseNetwork(state_space, action_space, layers, policy)
