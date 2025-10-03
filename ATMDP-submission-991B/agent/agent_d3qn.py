import torch
import copy
import torch.nn as nn
import numpy as np

from agent.zsc_agent import PursuerAgent
from prioritized_memory import Memory
from net.net_d3qn import ActorCriticD3QN


class AgentD3QN(PursuerAgent):
    """
    The D3QN agent.
    Class methods:
        __init__: the initialization function
        choose_action: choose action according to epsilon-greedy method
        store_transition: store transitions into replay memory
        learn: sample a batch of transitions from the replay memory and update parameters of the Q network
    """

    def __init__(self, config_args, idx, pretrain_net=None, ctrl_num=1):
        if pretrain_net is not None:
            super().__init__(pretrain_net, idx, ctrl_num)
        else:
            super().__init__(ActorCriticD3QN(config_args["num_action"]), idx, ctrl_num)
            # Hyper Parameters
            if config_args is not None:
                self.episode_max = config_args["episode_max"]  # the amount of episodes used to train
                self.batch_size = config_args["batch_size"]  # the size of the mini batch
                self.lr = config_args["learning_rate"]  # learning rate
                self.epsilon_origin = config_args["epsilon_origin"]  # original epsilon
                self.epsilon_decrement = config_args["epsilon_decrement"]  # epsilon decay
                self.gamma = config_args["gamma"]  # the discount factor
                self.target_replace_iter = config_args["target_replace_iter"]  # update frequency of target network

            self.target_net = ActorCriticD3QN(config_args["num_action"])  # target network
            self.learn_step_counter = 0  # the counter of update
            self.memory = Memory(config_args["memory_size"], config_args["num_state"])  # the prioritized replay memory
            self.max_td_error = 0.  # the maximal td error
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
            self.loss_func = nn.MSELoss(reduction='none')  # MSE loss
        self.num_evader = config_args["num_e"]
        self.num_agent = config_args["num_p"]

    def choose_action(self, state, epsilon, num_action, device):
        """
        Choose an action according to epsilon-greedy method.
        Input:
            state: the observation of pursuers
            epsilon: the value of current epsilon
        Output:
            action: the action index chosen
        """
        # observations
        state = torch.tensor(np.ravel(state), dtype=torch.float32, device=device).view(1, -1)
        # epsilon-greedy method
        if np.random.uniform() > epsilon:
            # choose the action with maximal Q value
            actions_value = self.eval_net(state, num_agent=self.num_agent, num_evader=self.num_evader)
            action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()

        else:
            # choose the action randomly
            action = np.random.randint(0, num_action)
        return action

    def store_transition(self, state, action, reward, state_next, done):
        """
        Store transitions into the replay memory.
        Input:
            state: the observation at this timestep
            action: the action executed at this timestep
            reward: the reward received from environment
            state_next: the observation at the next timestep
            done: whether the pursuer is inactive
        """
        # add transitions into replay memory
        transition = np.hstack(
            (np.ravel(state), np.ravel(action), np.ravel(reward), np.ravel(state_next), np.ravel(done)))
        self.memory.add(self.max_td_error, transition)

    def learn(self, i_episode, num_state, device):
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
        b_state = torch.tensor(b_memory[:, :num_state], dtype=torch.float32, device=device)
        b_action = torch.tensor(b_memory[:, num_state:num_state + 1], dtype=torch.int64, device=device)
        b_reward = torch.tensor(b_memory[:, num_state + 1:num_state + 2], dtype=torch.float32, device=device)
        b_state_next = torch.tensor(b_memory[:, num_state + 2:num_state * 2 + 2], dtype=torch.float32,
                                    device=device)
        b_done = torch.tensor(b_memory[:, num_state * 2 + 2:num_state * 2 + 3], dtype=torch.float32, device=device)
        # calculate Q values
        q_eval = self.eval_net(b_state, num_agent=self.num_agent, num_evader=self.num_evader).gather(1, b_action)
        q_next_target_net = self.target_net(b_state_next, num_agent=self.num_agent, num_evader=self.num_evader)
        q_next_eval_net = self.eval_net(b_state_next, num_agent=self.num_agent, num_evader=self.num_evader)
        q_target = b_reward + self.gamma * torch.abs(1 - b_done) * q_next_target_net.gather(1,
                                                                                            torch.argmax(
                                                                                                q_next_eval_net,
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
        # calculate loss
        loss = (self.loss_func(q_eval, q_target.detach()) * torch.FloatTensor(omega).to(device).detach()).mean()
        # back propagate
        self.optimizer.zero_grad()
        loss.backward()
        # update parameters
        self.optimizer.step()
        return loss.cpu().item()

    def cal_orientation_vector(self, env, choose_action, index=None, param_dic=None):
        if index is None:
            raise ValueError("The index of the agent is not given.")
        if param_dic is None:
            raise ValueError("The parameter dictionary is not given.")
        self.check_param_dic(param_dic)

        # extract parameters
        F_temp = param_dic["F_temp"]

        # calculate the orientation vector
        result_dict = {}
        agent_orientation = env.agent_orientation
        F_APF = copy.deepcopy(F_temp[:, index:index + 1])
        self_orientation = agent_orientation[:, index:index + 1]
        temp = np.radians(30)
        if np.arccos(np.clip(np.dot(np.ravel(self_orientation), np.ravel(F_APF)) / np.linalg.norm(
                self_orientation) / np.linalg.norm(F_APF), -1, 1)) > temp:
            rotate_matrix = np.array([[np.cos(temp), -np.sin(temp)], [np.sin(temp), np.cos(temp)]])
            temp1 = np.matmul(rotate_matrix, self_orientation)
            rotate_matrix = np.array(
                [[np.cos(-temp), -np.sin(-temp)], [np.sin(-temp), np.cos(-temp)]])
            temp2 = np.matmul(rotate_matrix, self_orientation)
            if np.dot(np.ravel(temp1), np.ravel(F_APF)) > np.dot(np.ravel(temp2), np.ravel(F_APF)):
                F_APF = temp1
            else:
                F_APF = temp2

        result_dict["F"] = copy.deepcopy(F_APF)
        return result_dict

    def check_param_dic(self, param_dic):
        if "F_temp" not in param_dic:
            raise ValueError("The parameter dictionary should contain the key F_temp.")
