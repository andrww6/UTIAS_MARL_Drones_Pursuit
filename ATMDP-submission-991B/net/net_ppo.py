import torch
import torch.nn as nn


class ActorCriticPPO(nn.Module):
    def __init__(self, action_dim):
        super(ActorCriticPPO, self).__init__()
        # actor
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(2, 128)
        self.layer3 = nn.Linear(2, 128)
        self.layer4 = nn.Linear(128 + 128 + 128, 128)
        self.layer5 = nn.Linear(128, 128)
        self.layer6 = nn.Linear(128, action_dim)
        self.logstd = nn.Parameter(torch.zeros((1, action_dim)))
        # critic
        self.layer7 = nn.Linear(2, 128)
        self.layer8 = nn.Linear(2, 128)
        self.layer9 = nn.Linear(2, 128)
        self.layer10 = nn.Linear(128 + 128 + 128, 128)
        self.layer11 = nn.Linear(128, 128)
        self.layer12 = nn.Linear(128, 1)

        torch.nn.init.orthogonal_(self.layer1.weight)
        torch.nn.init.orthogonal_(self.layer2.weight)
        torch.nn.init.orthogonal_(self.layer3.weight)
        torch.nn.init.orthogonal_(self.layer4.weight)
        torch.nn.init.orthogonal_(self.layer5.weight)
        torch.nn.init.orthogonal_(self.layer6.weight)
        torch.nn.init.orthogonal_(self.layer7.weight)
        torch.nn.init.orthogonal_(self.layer8.weight)
        torch.nn.init.orthogonal_(self.layer9.weight)
        torch.nn.init.orthogonal_(self.layer10.weight)
        torch.nn.init.orthogonal_(self.layer11.weight)
        torch.nn.init.orthogonal_(self.layer12.weight)

        torch.nn.init.constant_(self.layer1.bias, 0.0)
        torch.nn.init.constant_(self.layer2.bias, 0.0)
        torch.nn.init.constant_(self.layer3.bias, 0.0)
        torch.nn.init.constant_(self.layer4.bias, 0.0)
        torch.nn.init.constant_(self.layer5.bias, 0.0)
        torch.nn.init.constant_(self.layer6.bias, 0.0)
        torch.nn.init.constant_(self.layer7.bias, 0.0)
        torch.nn.init.constant_(self.layer8.bias, 0.0)
        torch.nn.init.constant_(self.layer9.bias, 0.0)
        torch.nn.init.constant_(self.layer10.bias, 0.0)
        torch.nn.init.constant_(self.layer11.bias, 0.0)
        torch.nn.init.constant_(self.layer12.bias, 0.0)

    def forward(self, x, num_agent, num_evader=None):
        state_target = x[:, :2*num_evader]
        state_obs = x[:, 2*num_evader:2*num_evader+2]
        state_ij = x[:, 2*num_evader+2:]
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

        input = torch.hstack((x_target, x_obs, x_ij))
        actor = torch.tanh(self.layer4(input))
        actor = torch.tanh(self.layer5(actor))
        mean = torch.tanh(self.layer6(actor))
        logstd = self.logstd.expand_as(mean)

        # critic
        x_ij_ori = state_ij.reshape(-1, 2)
        x_ij_ori = torch.tanh(self.layer7(x_ij_ori))

        x_ij = x_ij_ori[0::ij_num, :] * ij_weight
        for i in range(1, ij_num):
            x_ij += x_ij_ori[i::ij_num, :] * ij_weight

        x_target = state_target.reshape(-1, 2)
        x_target_ori = torch.tanh(self.layer8(x_target))
        
        target_wight = 1 / num_evader
        x_target = x_target_ori[0::num_evader, :] * target_wight
        for i in range(1, num_evader):
            x_target += x_target_ori[i::num_evader, :] * target_wight

        x_obs = torch.tanh(self.layer9(state_obs))

        input = torch.hstack((x_target, x_obs, x_ij))
        critic = torch.tanh(self.layer10(input))
        critic = torch.tanh(self.layer11(critic))
        state_value = self.layer12(critic)
        return mean, logstd, state_value