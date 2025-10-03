import os
import gymnasium as gym
import random
import torch
import numpy as np

from env.environment import environment


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def make_env(env_config):
    return environment(env_config)


class RunningStat:
    """
        Calculate the mean and std of all previous rewards.
    Class methods:
        __init__: the initialization function
        push: update statistics
    """

    def __init__(self):
        """
        The initialization function
        """
        self.n = 0  # the number of reward signals collected
        self.mean = np.zeros((1,))  # the mean of all rewards
        self.s = np.zeros((1,))
        self.std = np.zeros((1,))  # the std of all rewards

    def push(self, x):
        """
        Update statistics.
        Input:
            x: the reward signal
        """
        self.n += 1  # update the number of reward signals collected
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n  # update mean
            self.s = self.s + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.s / (self.n - 1) if self.n > 1 else np.square(self.mean))  # update stm


# def warm_up(train_agent, train_index, configs, memory_size, population, running_stat):
#     num_env = 8
#     # num_env = configs["num_envs"]
#     num_agent = configs["num_p"]
#     syn_train_env = gym.vector.SyncVectorEnv([
#         make_env(configs) for _ in range(num_env)
#     ])
#
#     while train_agent.memory.sumtree.n_entries < memory_size:
#         _ = syn_train_env.reset()
#
#         env_teammate_list = []
#         for env_index in range(num_env):
#             env_teammate_list.append(random_teammates(population, config, train_index))
#
#         while True:
#             last_done_array = np.array([np.zeros((1, num_agent)) for _ in
#                                         range(num_env)])  # whether pursuers are inactive at the last timestep
#             chooses_actions_array = np.array([np.zeros((1, num_agent)) for _ in range(num_env)])
#             state_array = syn_train_env.get_attr("state")
#
#             env_finished = False
#             for env_index in range(num_env):
#                 state = state_array[env_index]
#
#                 chooses_actions = chooses_actions_array[env_index]
#                 teammate_list = env_teammate_list[env_index]
#                 for i in range(num_agent):
#                     current_agent = train_agent
#                     if i != 0:
#                         current_agent = teammate_list[i - 1]
#                     state_temp = state[:, i:i + 1].reshape(1, -1)
#                     if isinstance(current_agent, AgentD3QN) or isinstance(current_agent, AgentDQN) or isinstance(
#                             current_agent, AgentDACOOP):
#                         action_temp = current_agent.choose_action(state_temp, max(
#                             config["epsilon_origin"] - config["epsilon_decrement"] * i_episode, 0.01),
#                                                                   train_env.num_action, device)
#                     elif isinstance(current_agent, AgentVicsek):
#                         action_temp = current_agent.choose_action(state_temp)
#                     elif isinstance(current_agent, AgentGreedy):
#                         action_temp = current_agent.choose_action(state_temp, train_env.agent_orientation,
#                                                                   train_env.num_agent)
#                     else:
#                         raise ValueError("The agent type is not supported.")
#
#                     chooses_actions[:, i] = action_temp
#                 # execute actions
#                 real_action = joint_action(actions=chooses_actions, agent_types=get_agent_type(teammate_list))
#                 chooses_actions_array[env_index] = real_action
#
#             _, _, _, _, infos = syn_train_env.step(chooses_actions_array)
#             state_next_array = syn_train_env.get_attr("state")
#             for env_index in range(num_env):
#                 s_state = state_array[env_index]
#                 s_state_next = state_next_array[env_index]
#
#                 if "final_info" in infos:
#                     infos = infos["final_info"]
#                     s_reward = infos[env_index]["all_rewards"]
#                     s_done = infos[env_index]["done"]
#                     s_action = infos[env_index]["choose_actions"]
#                 else:
#                     s_reward = infos["all_rewards"][env_index]
#                     s_done = infos["done"][env_index]
#                     s_action = infos["choose_actions"][env_index]
#
#                 for i in range(num_agent):
#                     if not np.ravel(last_done_array[env_index])[i]:
#                         running_stat.push(s_reward[:, i])
#                         s_reward[0, i] = np.clip(s_reward[0, i] / (running_stat.std + 1e-8), -10, 10)
#                         train_agent.store_transition(s_state[:, i:i + 1], s_action[:, i:i + 1],
#                                                      s_reward[:, i:i + 1],
#                                                      s_state_next[:, i:i + 1], s_done[:, i:i + 1])
#
#                 if np.all(s_done):
#                     env_finished = True
#                     break
#                 last_done_array[env_index] = s_done
#
#             if env_finished:
#                 break
#
#         print("warm up: ", train_agent.memory.sumtree.n_entries)
#         if train_agent.memory.sumtree.n_entries >= memory_size:
#             break
