import copy
import torch
import numpy as np
import torch.nn as nn

from prioritized_memory import Memory
from agent.zsc_agent import PursuerAgent
from net.net_dacoop import ActorCriticDACOOP


class AgentDACOOP(PursuerAgent):
    """
    The DACOOP agent.
    Class methods:
        __init__: the initialization function
        cal_orientation_vector: calculate the orientation vector of the pursuer
    """

    def __init__(self, config_args, idx, pretrain_net=None, ctrl_num=1):
        self.config = config_args
        if pretrain_net is not None:
            super().__init__(pretrain_net, idx, ctrl_num)
        else:
            super().__init__(ActorCriticDACOOP(config_args["num_action"], config_args["num_e"]), idx, ctrl_num)

            self.gamma = config_args["gamma"]
            self.lr = config_args["learning_rate"]
            self.kl_weight = config_args["kl_weight"]
            self.state_dim = config_args["num_state"]
            self.batch_size = config_args["batch_size"]
            self.episode_max = config_args["episode_max"]
            self.target_replace_iter = config_args["target_replace_iter"]

            self.target_net = ActorCriticDACOOP(config_args["num_action"], config_args["num_e"])
            self.learn_step_counter = 0
            self.memory = Memory(int(config_args["memory_size"]), self.state_dim)
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
            self.loss_func = nn.MSELoss(reduction='none')
            self.max_td_error = 0.

    def store_transition(self, state, action, reward, state_next, done):
        """
        Store the transition into the memory.
        Input:
            state: the current state
            action: the action taken
            reward: the reward received
            state_next: the next state
            done: whether the episode is done
        """
        transition = np.hstack(
            (np.ravel(state), np.ravel(action), np.ravel(reward), np.ravel(state_next), np.ravel(done)))
        self.memory.add(self.max_td_error, transition)

    def choose_action(self, state, epsilon):
        """
        Choose an action according to epsilon-greedy method.
        Input:
            state: the observation of pursuers
            epsilon: the value of current epsilon
        Output:
            action: the action index chosen
        """
        # observations
        state = torch.tensor(np.ravel(state), dtype=torch.float32, device=self.device).view(1, -1)
        # epsilon-greedy method
        if np.random.uniform() > epsilon:
            # choose the action with maximal Q value
            actions_value, attention_score = self.eval_net(state)
            action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()

        else:
            # choose the action randomly
            actions_value, attention_score = self.eval_net(state)
            action = np.random.randint(0, self.state_dim)
        return action, attention_score.detach().to('cpu').numpy()

    def learn(self, i_episode, device):
        """
        Sample a batch of transitions from replay memory and update parameters of the Q network.
        Input:
            i_episode: the index of the current episode
        """
        if self.learn_step_counter % self.target_replace_iter == 0:
            # periodically update the target network
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample transition batch
        b_memory, indexs, omega = self.memory.sample(self.batch_size, i_episode, self.episode_max)
        b_state = torch.tensor(b_memory[:, :self.state_dim], dtype=torch.float32, device=device)
        b_action = torch.tensor(b_memory[:, self.state_dim:self.state_dim + 1], dtype=torch.int64, device=device)
        b_reward = torch.tensor(b_memory[:, self.state_dim + 1:self.state_dim + 2], dtype=torch.float32,
                                device=device)
        b_state_next = torch.tensor(b_memory[:, self.state_dim + 2:self.state_dim * 2 + 2], dtype=torch.float32,
                                    device=device)
        b_done = torch.tensor(b_memory[:, self.state_dim * 2 + 2:self.state_dim * 2 + 3], dtype=torch.float32,
                              device=device)
        # calculate Q values
        temp1, attention_now = self.eval_net(b_state)
        q_eval = temp1.gather(1, b_action)
        with torch.no_grad():
            q_next_targetnet, _ = self.target_net(b_state_next)
        q_next_evalnet, _ = self.eval_net(b_state_next)
        q_target = b_reward + self.gamma * torch.abs(1 - b_done) * q_next_targetnet.gather(1,
                                                                                           torch.argmax(q_next_evalnet,
                                                                                                        axis=1,
                                                                                                        keepdim=True))
        # calculate td errors
        td_errors = (q_target - q_eval).to('cpu').detach().numpy().reshape((-1, 1))
        # update prioritized replay memory
        self.max_td_error = max(np.max(np.abs(td_errors)), self.max_td_error)
        for i in range(self.batch_size):
            index = indexs[i, 0]
            td_error = td_errors[i, 0]
            self.memory.update(index, td_error)
        ##########################
        with torch.no_grad():
            _, attention_old = self.target_net(b_state)
        attention_old = attention_old.detach()
        loss_kl = attention_old[:, 0:1] * torch.log(attention_old[:, 0:1] / attention_now[:, 0:1]) + \
                  attention_old[:, 1:2] * torch.log(attention_old[:, 1:2] / attention_now[:, 1:2])
        ##########################
        # calculate loss
        loss = (self.loss_func(q_eval, q_target.detach()) * torch.FloatTensor(omega).to(self.device).detach() + \
                loss_kl * self.kl_weight).mean()
        if not torch.isfinite(loss).all():
            print('loss is not finite')
        # back propagate
        self.optimizer.zero_grad()
        loss.backward()
        # update parameters
        self.optimizer.step()

        params = torch.cat([p.view(-1) for p in self.eval_net.parameters()])
        if not torch.isfinite(params).all():
            print('eval_net parameters are not finite')
