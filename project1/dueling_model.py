import torch
from torch import nn
import torch.nn.functional as F



class Dueling_QNetwork(nn.Module):
    """
    Implementation of dueling DQN architecture. See: https://arxiv.org/pdf/1511.06581.pdf
    """
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=128):
        """
        Initialize parameters and build model.
        :param state_size: Dimension of the states
        :param action_size: Dimension of the actions
        :param seed: seed for RNG
        :param fc1_units: First hidden layer node count
        :param fc2_units: Second hidden layer node count
        """
        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # value layer
        self.fc_v = nn.Linear(fc2_units, 1)
        # advantage layer
        self.fc_a = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        Forward function for the network.
        :param state: input state
        :return: Tensor output of the last hidden layer
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        a = self.fc_a(x)
        # Combine the value and advantage streams to final output.
        # Nomalized a with minus a.mean
        x = v + (a - a.mean(1).unsqueeze(1))
        return x