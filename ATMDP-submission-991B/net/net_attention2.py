import torch
import torch.nn as nn

class ZscAttention2(nn.Module):
    '''
    The network class, pytorch==1.8.
    class methods:
        __init__: initialization function, 6 fully-connected layers
        forward: forward propagation
    Ref:Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning.
    '''

    def __init__(self, num_action):
        super(ZscAttention2, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128 + 5, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, num_action)
        self.layer6 = nn.Linear(64, 1)
        self.key = nn.Linear(128, 64)
        self.attention = nn.Linear(64 + 2 + 2 + 1 + 128, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        Input:
            x: observations
        output:
            action_value: Q value for each action
        '''
        state_loc = x[:, :5]
        state_ij = x[:, 5:]
        # embedding
        x = state_ij.reshape(-1, 2)
        x = torch.relu(self.layer1(x))
        mean_embedding = (x[0::2, :] + x[1::2, :]) / 2
        mean_embedding = torch.hstack((mean_embedding, mean_embedding)).reshape(-1, mean_embedding.shape[1])
        # attention
        key = torch.relu(self.key(x))
        attention_input = torch.hstack((key, mean_embedding,
                                        torch.reshape(torch.hstack((state_loc, state_loc)),
                                                      (-1, state_loc.shape[1]))))
        attention_score = self.attention(attention_input)
        attention_score_normalized = self.softmax(torch.reshape(attention_score, (-1, 2))).reshape((-1, 1))
        temp = attention_score_normalized * x
        weighted_feature = temp[0::2, :] + temp[1::2, :]
        # concatenate
        network_input = torch.hstack((state_loc, weighted_feature))
        # calculate advantage
        x = torch.relu(self.layer2(network_input))
        advantage = torch.relu(self.layer3(x))
        advantage = self.layer5(advantage)
        # calculate state value
        state_value = torch.relu(self.layer4(x))
        state_value = self.layer6(state_value)
        # calculate Q value
        action_value = state_value + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return action_value, attention_score_normalized.detach().to('cpu').numpy().reshape(1, -1)
