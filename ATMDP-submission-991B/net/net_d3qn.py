import torch
import torch.nn as nn


class ActorCriticD3QN(nn.Module):
    """
    The network class, pytorch==1.8.
    Class methods:
        __init__: initialization function, 6 fully-connected layers
        forward: forward propagation
    Ref:Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning.
    """

    def __init__(self, num_action):
        super(ActorCriticD3QN, self).__init__()

        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(2, 128)
        self.layer3 = nn.Linear(2, 128)

        self.layer4 = nn.Linear(128 + 128 + 128, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(128, 64)
        self.layer7 = nn.Linear(64, num_action)
        self.layer8 = nn.Linear(64, 1)

        torch.nn.init.orthogonal_(self.layer1.weight)
        torch.nn.init.orthogonal_(self.layer2.weight)
        torch.nn.init.orthogonal_(self.layer3.weight)
        torch.nn.init.orthogonal_(self.layer4.weight)
        torch.nn.init.orthogonal_(self.layer5.weight)
        torch.nn.init.orthogonal_(self.layer6.weight)
        torch.nn.init.orthogonal_(self.layer7.weight)
        torch.nn.init.orthogonal_(self.layer8.weight)

        torch.nn.init.constant_(self.layer1.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)
        torch.nn.init.constant_(self.layer3.bias, 0.0)
        torch.nn.init.constant_(self.layer4.bias, 0.0)
        torch.nn.init.constant_(self.layer5.bias, 0.0)
        torch.nn.init.constant_(self.layer6.bias, 0.0)
        torch.nn.init.constant_(self.layer7.bias, 0.0)
        torch.nn.init.constant_(self.layer8.bias, 0.0)

    def forward(self, x, num_agent=None, num_evader=None):
        """
        Input:
            x: observations/states, [:, :5] is the location information, [:, 5:] is the embedding of location information.
        output:
            action_value: Q value for each action. The shape is (batch_size, num_action)
            In this case, num_action is 24.
            for each action_value, the value will be transformed to a pointed (λ|eta) pair, which will be used into APF Field.
        """
        # 2 * self.num_evader + 2 + 1 + (self.num_agent - 1) * 2
        # state_target = x[:, :4]
        # state_obs = x[:, 4:6]
        # state_ij = x[:, 6:]
        #
        # x_ij = state_ij.reshape(-1, 2)
        # x_ij = torch.tanh(self.layer1(x_ij))
        # x_ij = x_ij[0::2, :] * 0.5 + x_ij[1::2, :] * 0.5
        #
        # x_target = state_target.reshape(-1, 2)
        # x_target = torch.tanh(self.layer2(x_target))
        # x_target = x_target[0::2, :] * 0.5 + x_target[1::2, :] * 0.5
        #
        # x_obs = torch.tanh(self.layer3(state_obs))
        #
        # mean_feature = torch.hstack((x_target, x_obs, x_ij))
        # print(mean_feature.shape)
        # import pdb; pdb.set_trace()
        state_target = x[:, :2 * num_evader]
        state_obs = x[:, 2 * num_evader:2 * num_evader + 2]
        state_ij = x[:, 2 * num_evader + 2 + 1:]
        # actor
        x_ij_ori = state_ij.reshape(-1, 2)
        x_ij_ori = torch.tanh(self.layer1(x_ij_ori))

        ij_num = num_agent - 1
        ij_weight = 1 / ij_num
        x_ij = x_ij_ori[0::ij_num, :] * ij_weight
        for i in range(1, ij_num):
            x_ij += x_ij_ori[i::ij_num, :] * ij_weight

        x_target = state_target.reshape(-1, 2)
        x_target_ori = torch.tanh(self.layer2(x_target))

        target_wight = 1 / num_evader
        x_target = x_target_ori[0::num_evader, :] * target_wight
        for i in range(1, num_evader):
            x_target += x_target_ori[i::num_evader, :] * target_wight

        x_obs = torch.tanh(self.layer3(state_obs))

        mean_feature = torch.hstack((x_target, x_obs, x_ij))

        # calculate advantage
        x = torch.relu(self.layer4(mean_feature))
        advantage = torch.relu(self.layer5(x))
        advantage = self.layer7(advantage)
        # calculate state value
        state_value = torch.relu(self.layer6(x))
        state_value = self.layer8(state_value)
        # calculate Q value, shape:
        action_value = state_value + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return action_value
