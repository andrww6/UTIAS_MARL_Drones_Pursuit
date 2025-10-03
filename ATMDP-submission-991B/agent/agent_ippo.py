import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np

from agent.zsc_agent import PursuerAgent
from net.net_ppo import ActorCriticPPO


class AgentIPPO(PursuerAgent):
    def __init__(self, config_args, idx, pretrain_net=None, ctrl_num=1):
        self.config = config_args
        if pretrain_net is not None:
            super().__init__(pretrain_net, idx, ctrl_num)
        else:
            super().__init__(ActorCriticPPO(config_args["action_dim"]), idx, ctrl_num)

        self.batch_size = config_args["batch_size"]  # the size of the mini batch
        self.ppo_epoch = config_args["ppo_epoch"]  # the size of the mini batch
        self.c1 = config_args["c1"]  # Value function coefficient
        self.c2 = config_args["c2"]  # Entropy bonus coefficient
        self.gae_lambda = config_args["gae_lambda"]
        self.eps_clip = config_args["eps_clip"]
        self.lr = config_args["learning_rate"]  # learning rate
        self.gamma = config_args["gamma"]  # the discount factor
        self.minibatch_size = config_args["minibatch_size"]
        self.accumulated_reward_last = 1
        self.advantage_last = 1
        # self.eval_net.load_state_dict(torch.load('1000.pt'))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.MseLoss = nn.MSELoss()

    def store_transition(self, state, action, reward, state_next, done, memory_index):
        return None

    def choose_action(self, state, num_agent, device, if_probs=False):

        if len(state.shape) == 2 and state.shape[1] == 1:
            # no multi-process
            state = torch.FloatTensor(state.transpose()).to(device)
        elif len(state.shape) == 3:
            state = torch.FloatTensor(state).to(device)
            state = torch.squeeze(state, dim=-1)
        mean, logstd, _ = self.eval_net(state, num_agent)
        dist = Normal(mean, torch.exp(logstd))
        action = dist.sample().detach()
        if if_probs:
            log_probs = dist.log_prob(action)
            action_probs = torch.exp(log_probs).detach()
            return action.to('cpu').numpy(), action_probs.to('cpu').numpy()
        return action.to('cpu').numpy()

    def learn(self, transition_dict, params, training_step):
        """
        :param transition_dict: 训练用的BUFFER，存储对应的数据
        [
            "states": (:, state_dimension),
            "actions": (:, action_dimension),
            "rewards": (:, 1),
            "next_states": (:, state_dimension),
            "dones": (:, 1)
        ]
        :return:
        """
        ######################
        # attributes from environment
        env_num_drones = params["env_num_drones"]
        device = params["device"]

        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).squeeze(-1).to(device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float32).view(-1, 1).to(device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float32).view(-1, 1).to(device)

        states_next = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float32).squeeze(-1).to(device)
        done = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float32).view(-1, 1).to(device)

        ######################
        advantage, ref_value = self.calculate_advantage(states, states_next, env_num_drones, rewards, done, device)
        advantage = advantage / (torch.std(advantage) + 1e-5)
        ######################
        mean, logstd, _ = self.eval_net(states, env_num_drones)
        old_logprob, _ = self.calculate_logprob(mean, logstd, actions)
        ######################
        loss_actor_recorder = []
        loss_critic_recorder = []
        loss_entropy_recorder = []

        buffer_data_length = states.shape[0]
        for _ in range(int(self.ppo_epoch * buffer_data_length / self.minibatch_size)):
            index = np.random.choice(buffer_data_length, self.minibatch_size)
            b_state = states[index, :]
            b_action = actions[index, :]
            b_advantage = advantage[index, :].detach()
            b_ref_value = ref_value[index, :].detach()
            b_old_logprob = old_logprob[index, :].detach()
            #### critic loss ####
            b_mean, b_logstd, b_state_value = self.eval_net(b_state, env_num_drones)
            loss_critic = self.MseLoss(b_state_value, b_ref_value) / b_ref_value.std()
            #### actor loss ####
            b_logprob, b_entropy = self.calculate_logprob(b_mean, b_logstd, b_action)
            ratio = torch.exp(b_logprob - b_old_logprob)
            surr1 = ratio * b_advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantage
            loss_actor = -torch.mean(torch.min(surr1, surr2))
            ####### entropy loss ##########
            loss_entropy = -torch.mean(b_entropy)
            ######### total loss ##########
            loss = loss_actor + self.c1 * loss_critic + self.c2 * loss_entropy
            #################
            loss_actor_recorder.append(loss_actor.item())
            loss_critic_recorder.append(loss_critic.item())
            loss_entropy_recorder.append(loss_entropy.item())
            ############## update ##############
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.eval_net.parameters(), 0.5)
            self.optimizer.step()
            training_step += 1
        return np.mean(loss_actor_recorder), np.mean(loss_critic_recorder), np.mean(
            loss_entropy_recorder), training_step

    def calculate_advantage(self, state, state_next, num_agent, reward, done, device):
        _, _, state_value = self.eval_net(state, num_agent)
        _, _, state_value_next = self.eval_net(state_next, num_agent)
        ################
        state_value = np.ravel(state_value.detach().to('cpu').numpy())
        state_value_next = np.ravel(state_value_next.detach().to('cpu').numpy())
        reward = np.ravel(reward.to('cpu').numpy())
        done = np.ravel(done.to('cpu').numpy())
        ################
        result_accumulated_reward = list()
        result_advantage = list()
        result_td_target = list()
        for state_value_temp, state_next_value_temp, reward_temp, done_temp in zip(reversed(state_value),
                                                                                   reversed(state_value_next),
                                                                                   reversed(reward), reversed(done)):
            if done_temp == 1 or done_temp == 3:
                accumulated_reward = reward_temp
                td_error = reward_temp - state_value_temp
                advantage = td_error
            elif done_temp == 2:
                accumulated_reward = reward_temp + self.gamma * state_next_value_temp
                td_error = reward_temp + self.gamma * state_next_value_temp - state_value_temp
                advantage = td_error
            else:
                accumulated_reward = reward_temp + self.gamma * self.accumulated_reward_last
                td_error = reward_temp + self.gamma * state_next_value_temp - state_value_temp
                advantage = td_error + self.gamma * self.gae_lambda * self.advantage_last
            result_accumulated_reward.append(accumulated_reward)
            result_advantage.append(advantage)
            result_td_target.append(td_error + state_value_temp)
            self.accumulated_reward_last = accumulated_reward
            self.advantage_last = advantage
        # ref_value = torch.FloatTensor(list(reversed(result_accumulated_reward))).reshape(-1, 1).to(device)
        ref_value = torch.FloatTensor(list(reversed(result_td_target))).reshape(-1, 1).to(device)
        advantage = torch.FloatTensor(list(reversed(result_advantage))).reshape(-1, 1).to(device)
        return advantage, ref_value

    def calculate_logprob(self, mean, logstd, action):
        if torch.isnan(mean).any():
            print("here is NaN in mean")
            print(mean)
            mean = torch.nan_to_num(mean, nan=0.0)
        if torch.isnan(logstd).any():
            print("here is NaN in logstd")
            print(logstd)
            logstd = torch.nan_to_num(logstd, nan=0.0)
        dist = Normal(mean, torch.exp(logstd))
        logprob = dist.log_prob(action).sum(1, keepdim=True)
        entropy = dist.entropy().sum(1, keepdim=True)
        return logprob, entropy

    def need_more_data(self):
        if len(self.memory0) > self.batch_size or len(self.memory1) > self.batch_size or len(
                self.memory2) > self.batch_size:
            return False
        return True
