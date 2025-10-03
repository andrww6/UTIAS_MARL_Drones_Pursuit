import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np

from agent.zsc_agent import PursuerAgent
from net.net_ppo import ActorCriticPPO


class AgentPPO(PursuerAgent):
    def __init__(self, config_args, idx, pretrain_net=None, ctrl_num=1):
        self.config = config_args
        if pretrain_net is not None:
            super().__init__(pretrain_net, idx, ctrl_num)
        else:
            super().__init__(ActorCriticPPO(config_args["action_dim"]), idx, ctrl_num)

        self.num_evader = config_args["num_e"]
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
        self.memory0 = []
        self.memory1 = []
        self.memory2 = []

    def store_transition(self, state, action, reward, state_next, done, memory_index):
        transition = np.hstack(
            (np.ravel(state), np.ravel(action), np.ravel(reward), np.ravel(state_next), np.ravel(done)))
        if memory_index == 0:
            self.memory0.append(transition)
        if memory_index == 1:
            self.memory1.append(transition)
        if memory_index == 2:
            self.memory2.append(transition)

    def choose_action(self, state, num_agent, device, if_probs=False):

        if len(state.shape) == 2 and state.shape[1] == 1:
            # no multi-process
            state = torch.FloatTensor(state.transpose()).to(device)
        elif len(state.shape) == 3:
            state = torch.FloatTensor(state).to(device)
            state = torch.squeeze(state, dim=-1)
        mean, logstd, _ = self.eval_net(state, num_agent, num_evader=self.num_evader)
        dist = Normal(mean, torch.exp(logstd))
        action = dist.sample().detach()
        if if_probs:
            log_probs = dist.log_prob(action)
            action_probs = torch.exp(log_probs).detach()
            return action.to('cpu').numpy(), action_probs.to('cpu').numpy()
        return action.to('cpu').numpy()

    def learn(self, num_state, action_dim, num_agent, device, training_step):
        ######################
        if len(self.memory1) == 0:
            memory = np.array(self.memory0)
        else:
            memory = np.vstack((np.array(self.memory0), np.array(self.memory1), np.array(self.memory2)))

        state = torch.FloatTensor(memory[:, :num_state]).to(device)
        action = torch.FloatTensor(memory[:, num_state:num_state + action_dim]).to(device)
        reward = memory[:, num_state + action_dim:num_state + action_dim + 1]
        state_next = torch.FloatTensor(
            memory[:, num_state + action_dim + 1:num_state * 2 + action_dim + 1]).to(device)
        done = memory[:, num_state * 2 + action_dim + 1:num_state * 2 + action_dim + 2]
        ######################
        advantage, ref_value = self.calculate_advantage(state, state_next, num_agent, reward, done, device)
        advantage = advantage / (torch.std(advantage) + 1e-5)
        ######################
        mean, logstd, _ = self.eval_net(state, num_agent, num_evader=self.num_evader)
        old_logprob, _ = self.calculate_logprob(mean, logstd, action)
        ######################
        loss_actor_recorder = []
        loss_critic_recorder = []
        loss_entropy_recorder = []

        for _ in range(int(self.ppo_epoch * memory.shape[0] / self.minibatch_size)):
            index = np.random.choice(memory.shape[0], self.minibatch_size)
            b_state = state[index, :]
            b_action = action[index, :]
            b_advantage = advantage[index, :].detach()
            b_ref_value = ref_value[index, :].detach()
            b_old_logprob = old_logprob[index, :].detach()
            #### critic loss ####
            b_mean, b_logstd, b_state_value = self.eval_net(b_state, num_agent, num_evader=self.num_evader)
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
        self.memory0 = []
        self.memory1 = []
        self.memory2 = []
        return np.mean(loss_actor_recorder), np.mean(loss_critic_recorder), np.mean(
            loss_entropy_recorder), training_step

    def calculate_advantage(self, state, state_next, num_agent, reward, done, device):
        _, _, state_value = self.eval_net(state, num_agent, num_evader=self.num_evader)
        _, _, state_value_next = self.eval_net(state_next, num_agent, num_evader=self.num_evader)
        ################
        state_value = np.ravel(state_value.detach().to('cpu').numpy())
        state_value_next = np.ravel(state_value_next.detach().to('cpu').numpy())
        reward = np.ravel(reward)
        done = np.ravel(done)
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
        if len(self.memory1) ==0:
            if len(self.memory0) > self.batch_size*self.config["num_p"]:
                return False
            return True
        else:
            if len(self.memory0) > self.batch_size or len(self.memory1) > self.batch_size or len(
                    self.memory2) > self.batch_size:
                return False
            return True
