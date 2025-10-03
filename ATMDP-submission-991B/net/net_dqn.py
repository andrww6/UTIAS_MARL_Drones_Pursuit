import torch
import torch.nn as nn


class ActorCriticDQN(nn.Module):
    """
    The network class, pytorch==1.8.
    Class methods:
        __init__: initialization function, 6 fully-connected layers
        forward: forward propagation
    Ref:Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning.
    """

    def __init__(self, num_action):
        super(ActorCriticDQN, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128 + 4, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, num_action)

    def forward(self, x):
        """
        Input:
            x: observations
        output:
            action_value: Q value for each action
        """
        state_loc = x[:, :4]
        # embedding
        state_ij = x[:, 4:]
        x = state_ij.reshape(-1, 2)
        x = torch.relu(self.layer1(x))
        feature = torch.vstack((x[0::2, :].reshape(1, -1), x[1::2, :].reshape(1, -1)))
        mean_feature = torch.nanquantile(feature, 0.5, dim=0).reshape(-1, x.shape[1])  # calculate mean of features
        # concatenate
        mean_feature = torch.hstack((state_loc, mean_feature))
        # calculate advantage
        x = torch.relu(self.layer2(mean_feature))
        x = torch.relu(self.layer3(x))
        action_value = self.layer4(x)
        return action_value