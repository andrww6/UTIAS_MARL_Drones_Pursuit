from utils.agent_util import get_single_agent_type, get_agent_type, eval_random_teammates
from agent.agent_type import AgentType
from env.environment import environment
from torch.distributions import Normal
from agent.agent_greedy import AgentGreedy
from agent.agent_vicsek import AgentVicsek
import os
import copy
import tqdm
import torch
import random
from tqdm import tqdm
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
from utils.run_util import make_env
from env.parall import ParallelEnv
import time


def validate_p_env(env_config, valid_agent, teammate_list, eval_nums, val_save_path, if_save=False, if_render=False,
                   print_info=True, device="cuda"):
    # create the environment
    start_time = time.time()
    if (valid_agent.eval_net is not None):
        valid_agent.eval_net.eval()

    env_num = 10
    env_instances = [make_env(env_config) for _ in range(env_num)]
    env = ParallelEnv(env_instances)

    ctrl_num = env_config["ctrl_num"]

    t_per_episode = list()
    is_collision = list()
    acc_num = 0
    rew = 0
    avg_time = 0
    num_agent = env_config["num_p"]
    ctrl_num = env_config["ctrl_num"]
    iters = int(eval_nums / env_num)
    count = 0
    if env_config['eval_method'] == 'zsc':
        teammates_list_val = np.random.choice(teammate_list, env_config["num_p"] - ctrl_num)
    elif env_config['eval_method'] == 'fcp':
        teammates_list_val = [valid_agent] * (env_config["num_p"] - ctrl_num)
    else:
        teammates_list_val = teammate_list

    agent_type = get_agent_type(teammates_list_val)
    agent_type_array = [agent_type] * env_num
    state, global_state, info = env.reset()  # reset the environment
    first_last = True
    last_state_array = state.copy()
    while True:
        # choose action
        chooses_actions = np.zeros((env_num, num_agent))
        last_chooses_actions_array = torch.tensor(np.zeros((env_num, env.num_agent)),
                                                  dtype=torch.float32, device=device)
        for i in range(num_agent):
            if i < ctrl_num:
                current_agent = valid_agent
                current_type = get_agent_type([current_agent])[0].name
            else:
                current_agent = teammates_list_val[i - ctrl_num]
                current_type = agent_type[i - ctrl_num].name

            if current_type in ['PPO']:
                first_temp = state[:, :, i:i + 1]
                if len(first_temp.shape) == 2 and first_temp.shape[1] == 1:
                    # no multi-process
                    first_temp = torch.FloatTensor(first_temp.transpose()).to(device)
                elif len(first_temp.shape) == 3:
                    first_temp = torch.FloatTensor(first_temp).to(device)
                    first_temp = torch.squeeze(first_temp, dim=-1)
                mean, logstd, _ = current_agent.eval_net(first_temp, env_config['num_p'], env_config['num_e'])
                # dist = Normal(mean, torch.exp(logstd))
                # action = np.clip(dist.sample().detach().to('cpu').numpy().transpose(), -1, 1)
                action = mean.detach().to('cpu').numpy()
                single_action_temp = action[:, 0]
                # print(single_action_temp.shape)

            elif current_type in ['POAM']:
                first_temp = state[:, i:i + 1]
                if env_config["num_e"] == 1:
                    new_array = np.array([[2], [0]])
                    first_temp = np.insert(first_temp, 2, new_array, axis=0)
                # net_actions = current_agent.choose_action(first_temp,'cpu')
                first_temp = torch.FloatTensor(first_temp.transpose()).to(device)
                last_state_temp = torch.FloatTensor(last_state_array[:, i:i + 1].transpose()).to(device)
                mean, logstd, _, _, _ = current_agent.eval_net(first_temp, None, env_config['num_p'],
                                                               env_config['num_e'],
                                                               last_chooses_actions_array[:, i:i + 1],
                                                               last_state_temp)
                action = mean.detach().to('cpu').numpy().transpose()
                single_action_temp = action[0][0]
            elif current_type in ['VICSEK', 'GREEDY']:
                single_action_temp = 0
            else:
                raise ValueError("Invalid algorithm!, current valid_algorithm is {}".format(env.algorithm))
            chooses_actions[:, i] = single_action_temp

        # execute action
        if first_last:
            first_last = False
        else:
            last_state_array = state.copy()
        state, _, rewards, dones, info = env.step(chooses_actions, agent_type_array, ctrl_num=ctrl_num)

        last_chooses_actions_array = torch.tensor(
            chooses_actions, device=last_chooses_actions_array.device
        )
        for env_idx in range(env_num):
            rew += rewards[env_idx][0]
            done = dones[env_idx]
            if np.all(done):
                count += 1
                if np.all(done == 1):
                    t_per_episode.append(env_instances[env_idx].t)
                    avg_time += env_instances[env_idx].t
                    is_collision.append(0)
                    acc_num += 1
                elif np.any(done == 3):
                    t_per_episode.append(2000)
                    is_collision.append(1)
                else:
                    t_per_episode.append(1000)
                    is_collision.append(0)
                state_inv, global_state_inv, _ = env.reset_ind(env_idx)
                last_done_inv = np.zeros((1, num_agent))
                state[env_idx] = state_inv
                global_state[env_idx] = global_state_inv
                dones[env_idx] = last_done_inv
        if count > eval_nums:
            break
    t_per_episode.append(-1)
    if acc_num == 0:
        avg_len = 0
    else:
        avg_len = avg_time / acc_num
    message = "{}/{} --- Evaluation Finished with acc rate {}%: collision rate:{}% Avg Ep Rew: {} Avg time len: {} Duration: {}......".format(
        count, eval_nums,
        acc_num / (count) * 100,
        np.sum(
            is_collision) / (count) * 100,
        rew / (count),
        avg_len,
        round(time.time() - start_time, 2)
    )
    if print_info:
        print(message)
    env.close()

    if if_save:
        if name == "vicsek" or name == "greedy":
            val_save_path = "eval_weights"
        os.makedirs(os.path.join(val_save_path, "res/"), exist_ok=True)
        np.savetxt(os.path.join(val_save_path, "res/", f'time.txt'),
                   t_per_episode)
        np.savetxt(os.path.join(val_save_path, "res/", f'collision.txt'),
                   is_collision)

    if (valid_agent.eval_net is not None):
        valid_agent.eval_net.train()
    return acc_num / count * 100, np.sum(is_collision) / count * 100, rew / count, message


def validate_n_env(env_config, drone_team, eval_nums, val_save_path, if_save=False, if_render=False,
                   print_info=True, name='', device="cuda:0"):
    # create the environment
    start_time = time.time()
    env = environment(env_config)
    env.mode = 'Valid'

    for s_a in drone_team:
        if s_a.eval_net.training:
            s_a.eval_net.eval()
        s_a.eval_net.to(device)
    teammates_list_val = copy.deepcopy(drone_team)

    t_per_episode = list()
    is_collision = list()
    # load parameters
    # net.to(device)
    # print("Evaluation Starting ......")
    acc_num = 0
    rew = 0
    avg_time = 0
    # for num_episode in tqdm(range(eval_nums)):
    for num_episode in range(eval_nums):
        state, _, info = env.reset()  # reset the environment
        j = 0
        fig_paths = []
        while True:
            if if_render and num_episode < 3:
                assert val_save_path != None
                episode_fig_path = os.path.join(val_save_path, "renders/figs/", str(num_episode))
                os.makedirs(episode_fig_path, exist_ok=True)
                j += 1
                env.render()  # render the pursuit process
                fig_path = os.path.join(episode_fig_path, f"{j}.png")
                fig_paths.append(fig_path)
                plt.savefig(fig_path)  # save figures
            # initialize some arguments
            action = np.zeros((1, 0))  # action index buffer

            # choose actions
            chooses_actions = np.zeros((1, env.num_agent))
            agent_type = get_agent_type(teammates_list_val)
            for i in range(env.num_agent):
                current_agent = teammates_list_val[i]
                current_type = agent_type[i].name
                if current_type in ['PPO', 'IPPO']:
                    first_temp = state[:, i:i + 1]
                    if env_config["num_e"] == 1:
                        new_array = np.array([[2], [0]])
                        first_temp = np.insert(first_temp, 2, new_array, axis=0)
                    # net_actions = current_agent.choose_action(first_temp,'cpu')
                    first_temp = torch.FloatTensor(first_temp.transpose()).to(device)
                    mean, logstd, _ = current_agent.eval_net(first_temp, env_config['num_p'])
                    # dist = Normal(mean, torch.exp(logstd))
                    # action = np.clip(dist.sample().detach().to('cpu').numpy().transpose(), -1, 1)
                    action = mean.detach().cpu().numpy().transpose()
                    single_action_temp = action[0][0]
                elif current_type in ['DQN', 'D3QN']:
                    c_state = info['d3qn_state']
                    if env_config["num_e"] == 1:
                        new_array = np.array([[2], [0]])
                        c_state = np.insert(c_state, 4, new_array, axis=0)

                    if np.all(info["capture"] == 0):
                        c_state = np.vstack((c_state[0:4, :], c_state[6:, :]))
                        first_temp = c_state[:, i:i + 1].reshape(1, -1)
                        state_tensor = torch.tensor(np.ravel(first_temp), dtype=torch.float32).view(1, -1)

                        first_temp = current_agent.eval_net(state_tensor)
                        first_action_temp = torch.max(first_temp, 1)[1].data.to('cpu').numpy()
                        single_action_temp = first_action_temp[0]
                    else:
                        single_action_temp = 0
                        teammates_list_val[i - 1] = AgentGreedy(idx=current_agent.idx)
                elif current_type in ['VICSEK', 'GREEDY']:
                    single_action_temp = 0
                else:
                    raise ValueError("Invalid algorithm!, current valid_algorithm is {}".format(env.algorithm))
                chooses_actions[:, i] = single_action_temp

            # execute action
            state, _, reward, done, info = env.step(chooses_actions, get_agent_type(teammates_list_val))
            rew += reward[0]

            if np.all(done):
                if np.all(done == 1):
                    t_per_episode.append(env.t)
                    avg_time += env.t
                    is_collision.append(0)
                    acc_num += 1
                elif np.any(done == 3):
                    t_per_episode.append(2000)
                    is_collision.append(1)
                else:
                    t_per_episode.append(1000)
                    is_collision.append(0)
                break
        if if_render and num_episode < 3:
            images = []
            for filename in fig_paths:
                images.append(imageio.imread(filename))
            os.makedirs(os.path.join(val_save_path, "renders/gifs/"), exist_ok=True)
            gif_path = os.path.join(val_save_path, "renders/gifs/", str(num_episode) + ".gif")
            imageio.mimsave(gif_path, images)
        if acc_num == 0:
            avg_len = 0
        else:
            avg_len = avg_time / acc_num
        count = num_episode + 1
        r_message = ("{}/{} --- Evaluation Finished with acc rate {}%: collision rate:{}% Avg Ep Rew: {} Avg time "
                     "len: {} Duration: {}......").format(
            count, eval_nums, acc_num / count * 100, np.sum(is_collision) / count * 100, rew / count, avg_len,
            round(time.time() - start_time, 2))
        if print_info and count % 20 == 0:
            print(r_message)

    if if_save:
        np.savetxt(os.path.join(val_save_path, f'time_{name}.txt'),
                   t_per_episode)
        np.savetxt(os.path.join(val_save_path, f'collision_{name}.txt'),
                   is_collision)

    for s_a in drone_team:
        if not s_a.eval_net.training:
            s_a.eval_net.train()
    print(r_message)
    return acc_num, np.sum(is_collision), rew / eval_nums


def validate_s_env(env_config, valid_agent, teammate_list, eval_nums, device, val_save_path, if_save=False,
                   if_render=False,
                   print_info=True, name=''):
    # create the environment
    # import pdb; pdb.set_trace()
    start_time = time.time()
    if (valid_agent.eval_net is not None):
        valid_agent.eval_net.eval()
    env = environment(env_config)
    env.mode = 'Valid'
    ctrl_num = env_config["ctrl_num"]
    t_per_episode = list()
    is_collision = list()
    # load parameters
    # net.to(device)
    # print("Evaluation Starting ......")
    acc_num = 0
    rew = 0
    avg_time = 0
    # for num_episode in tqdm(range(eval_nums)):
    for num_episode in range(eval_nums):
        if env_config['eval_method'] == 'zsc':
            teammates_list_val = np.random.choice(teammate_list, env_config["num_p"] - ctrl_num)
        elif env_config['eval_method'] == 'fcp':
            teammates_list_val = [valid_agent] * (env_config["num_p"] - 1)
        else:
            teammates_list_val = teammate_list
        state, global_state, info = env.reset()  # reset the environment
        last_state = copy.deepcopy(state)
        j = 0
        fig_paths = []
        while True:
            if if_render and num_episode < 3:
                assert val_save_path != None
                episode_fig_path = os.path.join(val_save_path, "renders/figs/", str(num_episode))
                os.makedirs(episode_fig_path, exist_ok=True)
                j += 1
                env.render()  # render the pursuit process
                fig_path = os.path.join(episode_fig_path, f"{j}.png")
                fig_paths.append(fig_path)
                plt.savefig(fig_path)  # save figures
            # initialize some arguments

            # choose actions
            chooses_actions = np.zeros((1, env.num_agent))
            agent_type = get_agent_type(teammates_list_val)
            for i in range(env.num_agent):
                current_agent = valid_agent
                current_type = env.algorithm
                if i > ctrl_num - 1:
                    current_agent = teammates_list_val[i - ctrl_num]
                    current_type = agent_type[i - ctrl_num].name
                if current_type in ['PPO']:
                    first_temp = state[:, i:i + 1]
                    if env_config["num_e"] == 1:
                        new_array = np.array([[2], [0]])
                        first_temp = np.insert(first_temp, 2, new_array, axis=0)
                    # net_actions = current_agent.choose_action(first_temp,'cpu')
                    first_temp = torch.FloatTensor(first_temp.transpose()).to(device)
                    mean, logstd, _ = current_agent.eval_net(first_temp, env_config['num_p'], env_config['num_e'])
                    # dist = Normal(mean, torch.exp(logstd))
                    # action = np.clip(dist.sample().detach().to('cpu').numpy().transpose(), -1, 1)
                    action = mean.detach().to('cpu').numpy().transpose()
                    single_action_temp = action[0][0]
                elif current_type in ['POAM']:
                    first_temp = state[:, i:i + 1]
                    first_last_temp = last_state[:, i:i + 1]
                    if env_config["num_e"] == 1:
                        new_array = np.array([[2], [0]])
                        first_temp = np.insert(first_temp, 2, new_array, axis=0)
                        first_last_temp = np.insert(first_last_temp, 2, new_array, axis=0)
                    # net_actions = current_agent.choose_action(first_temp,'cpu')
                    first_temp = torch.FloatTensor(first_temp.transpose()).to(device)
                    first_last_temp = torch.FloatTensor(first_last_temp.transpose()).to(device)
                    mean, logstd, _ = current_agent.eval_net(first_temp, None, first_last_temp, env_config['num_p'],
                                                             env_config['num_e'])
                    # dist = Normal(mean, torch.exp(logstd))
                    # action = np.clip(dist.sample().detach().to('cpu').numpy().transpose(), -1, 1)
                    action = mean.detach().to('cpu').numpy().transpose()
                    single_action_temp = action[0][0]
                elif current_type in ["MAPPO"]:
                    first_temp = state[:, i:i + 1]
                    if env_config["num_e"] == 1:
                        new_array = np.array([[2], [0]])
                        first_temp = np.insert(first_temp, 2, new_array, axis=0)
                    # net_actions = current_agent.choose_action(first_temp,'cpu')
                    first_temp = torch.FloatTensor(first_temp.transpose()).to(device)
                    mean, logstd, _ = current_agent.eval_net(first_temp, None, env_config['num_p'], env_config['num_e'])
                    # dist = Normal(mean, torch.exp(logstd))
                    # action = np.clip(dist.sample().detach().to('cpu').numpy().transpose(), -1, 1)
                    action = mean.detach().to('cpu').numpy().transpose()
                    single_action_temp = action[0][0]
                elif current_type in ['DQN', 'D3QN']:
                    c_state = info['d3qn_state']
                    if env_config["num_e"] == 1:
                        new_array = np.array([[2], [0]])
                        c_state = np.insert(c_state, 4, new_array, axis=0)

                    if np.all(info["capture"] == 0):
                        c_state = np.vstack((c_state[0:4, :], c_state[6:, :]))
                        # elif info["capture"][0][0] == 1:
                        #     c_state = np.vstack((c_state[0:2,:], c_state[4:,:]))
                        # elif info["capture"][0][1] == 1:
                        #     c_state = np.vstack((c_state[0:4,:], c_state[6:,:]))
                        # else:
                        #     print(info["capture"])
                        first_temp = c_state[:, i:i + 1].reshape(1, -1)
                        state_tensor = torch.tensor(np.ravel(first_temp), dtype=torch.float32).view(1, -1)

                        first_temp = current_agent.eval_net(state_tensor)
                        first_action_temp = torch.max(first_temp, 1)[1].data.to('cpu').numpy()
                        # single_action = np.hstack((action, np.array(first_action_temp, ndmin=2)))
                        # single_action_temp = copy.deepcopy(single_action)
                        single_action_temp = first_action_temp[0]
                    else:
                        single_action_temp = 0
                        teammates_list_val[i - ctrl_num] = AgentGreedy(idx=current_agent.idx)
                elif current_type in ['VICSEK', 'GREEDY']:
                    single_action_temp = 0
                else:
                    raise ValueError("Invalid algorithm!, current valid_algorithm is {}".format(env.algorithm))
                chooses_actions[:, i] = single_action_temp

            # execute action
            next_state, _, reward, done, info = env.step(chooses_actions, get_agent_type(teammates_list_val),
                                                         ctrl_num=ctrl_num)

            last_state = state
            state = next_state

            rew += reward[0]

            if np.all(done):
                if np.all(done == 1):
                    t_per_episode.append(env.t)
                    avg_time += env.t
                    is_collision.append(0)
                    acc_num += 1
                elif np.any(done == 3):
                    t_per_episode.append(2000)
                    is_collision.append(1)
                else:
                    t_per_episode.append(1000)
                    is_collision.append(0)
                break
        if if_render and num_episode < 3:
            images = []
            for filename in fig_paths:
                images.append(imageio.imread(filename))
            os.makedirs(os.path.join(val_save_path, "renders/gifs/"), exist_ok=True)
            gif_path = os.path.join(val_save_path, "renders/gifs/", str(num_episode) + ".gif")
            imageio.mimsave(gif_path, images)
        if acc_num == 0:
            avg_len = 0
        else:
            avg_len = avg_time / acc_num
        count = num_episode + 1
        message = "{}/{} --- Evaluation Finished with acc rate {}%: collision rate:{}% Avg Ep Rew: {} Avg time len: {} Duration: {}......".format(
            count, eval_nums,
            acc_num / (count) * 100,
            np.sum(
                is_collision) / (count) * 100,
            rew / (count),
            avg_len,
            round(time.time() - start_time, 2)
        )
        if print_info and count % 10 == 0:
            print(message)

    if if_save:
        np.savetxt(os.path.join(val_save_path, f'time_{name}.txt'),
                   t_per_episode)
        np.savetxt(os.path.join(val_save_path, f'collision_{name}.txt'),
                   is_collision)
    if (valid_agent.eval_net is not None):
        valid_agent.eval_net.train()
    print(message)
    return acc_num, np.sum(is_collision), rew / eval_nums, message


def eval_validate(configs, valid_agent, teammate_candidates, device, val_save_path, if_save=False,
                  if_render=False, print_info=True):
    # set valid_agent model to eval mode
    if valid_agent.eval_net is not None:
        valid_agent.eval_net.eval()
    # create the environment
    env = environment(configs)
    env.mode = 'Valid'
    assert env.num_evader > 1

    start_time = time.time()
    ctrl_num = valid_agent.ctrl_num

    # 验证传进来的队友候选者是否符合规范，如果是直接验证，那么需要在验证脚本调用这个方法前通过config把候选者给加载好
    assert teammate_candidates is not None
    candidates_ctrl_num = 0
    for i in range(len(teammate_candidates)):
        candidates_ctrl_num += teammate_candidates[0].ctrl_num
    assert candidates_ctrl_num >= env.num_agent - valid_agent.ctrl_num

    eval_nums = configs["eval_nums"]
    # valid recoder
    t_per_episode = list()
    is_collision = list()
    acc_num = 0
    rew = 0
    avg_time = 0
    # print("Evaluation Starting ......")
    for num_episode in range(eval_nums):
        # pick teammates by difference eval method
        teammates = eval_random_teammates(valid_agent, teammate_candidates, env.num_agent - valid_agent.ctrl_num,
                                          configs["eval_method"])
        ori_agent_team_p1 = [valid_agent] * ctrl_num
        ori_agent_team_p2 = []
        for i in range(len(teammates)):
            ori_agent_team_p2.extend([teammates[i]] * teammates[i].ctrl_num)

        mates_type = get_agent_type(ori_agent_team_p2)
        current_agent_team = ori_agent_team_p1 + ori_agent_team_p2
        assert len(current_agent_team) == env.num_agent

        # reset valid environment
        state, global_state, info = env.reset()  # reset the environment

        # fig params
        j = 0
        fig_paths = []

        # special params for different agent
        # poam
        last_state = copy.deepcopy(state)
        # dacoop
        attention_score_array = np.zeros((env.num_agent, env.num_agent - 1))

        while True:
            if if_render and num_episode < 3:
                assert val_save_path is not None
                episode_fig_path = os.path.join(val_save_path, "renders/figs/", str(num_episode))
                os.makedirs(episode_fig_path, exist_ok=True)
                j += 1
                env.render()  # render the pursuit process
                fig_path = os.path.join(episode_fig_path, f"{j}.png")
                fig_paths.append(fig_path)
                plt.savefig(fig_path)  # save figures

            # choose actions temp
            chooses_actions = np.zeros((1, env.num_agent))

            for i in range(len(current_agent_team)):
                current_agent = current_agent_team[i]
                current_type = get_single_agent_type(current_agent)
                # MAPPO, D3QN的ctrl_num只能为1
                if current_type == AgentType.D3QN or current_type == AgentType.MAPPO:
                    assert current_agent.ctrl_num == 1
                if current_type == AgentType.PPO:
                    state_temp = state[:, :i + 1]
                    state_temp = torch.FloatTensor(state_temp.transpose()).to(device)
                    mean, logstd, _ = current_agent.eval_net(state_temp, env.num_agent, env.num_evader)
                    action = mean.detach().to('cpu').numpy().transpose()
                    single_action_temp = action[0][0]
                elif current_type == AgentType.POAM:
                    state_temp = state[:, i:i + 1]
                    state_temp = torch.FloatTensor(state_temp.transpose()).to(device)
                    last_state_temp = torch.FloatTensor(last_state[:, i:i + 1].transpose()).to(device)
                    mean, logstd, _, _, _ = current_agent.eval_net(state_temp, None, last_state_temp, env.num_agent,
                                                                   env.num_evader)
                    action = mean.detach().to('cpu').numpy().transpose()
                    single_action_temp = action[0][0]
                elif current_type == AgentType.MAPPO:
                    state_temp = state[:, i:i + 1]
                    state_temp = torch.FloatTensor(state_temp.transpose()).to(device)
                    mean, logstd, _ = current_agent.eval_net(state_temp, None, env.num_agent,
                                                             env.num_evader)
                    action = mean.detach().to('cpu').numpy().transpose()
                    single_action_temp = action[0][0]
                elif current_type == AgentType.DACOOP:
                    c_state = info['d3qn_state']
                    # choose actions
                    temp = c_state[:, i:i + 1].reshape(1, -1)
                    temp, attention_score = current_agent.eval_net(
                        torch.tensor(np.ravel(temp), dtype=torch.float32).view(1, -1))
                    action_temp = torch.max(temp, 1)[1].data.to('cpu').numpy()
                    single_action_temp = action_temp[0]
                    attention_score_array[i:i + 1, :] = attention_score.detach().numpy()
                elif current_type == AgentType.D3QN or current_type == AgentType.DQN:
                    c_state = info['d3qn_state']
                    # D3QN和DQN如果抓住了第一个，后面再用Greedy的行为来操作
                    if np.all(info["capture"] == 0):
                        c_state = np.vstack((c_state[0:4, :], c_state[6:, :]))
                        c_state_temp = c_state[:, i:i + 1].reshape(1, -1)
                        state_tensor = torch.tensor(np.ravel(c_state_temp), dtype=torch.float32).view(1, -1)

                        c_state_temp = current_agent.eval_net(state_tensor)
                        c_action_temp = torch.max(c_state_temp, 1)[1].data.to('cpu').numpy()
                        single_action_temp = c_action_temp[0]
                    else:
                        single_action_temp = 0
                        current_agent_team[i] = AgentGreedy(idx=current_agent.idx)
                elif current_type == AgentType.VICSEK or current_type == AgentType.GREEDY:
                    single_action_temp = 0
                else:
                    raise ValueError("Invalid algorithm!, current valid_algorithm is {}".format(env.algorithm))
                chooses_actions[:, i] = single_action_temp

            # execute action
            next_state, _, reward, done, info = env.step(chooses_actions, mates_type, ctrl_num=ctrl_num)

            # special params for different agent
            # poam

            last_state = state
            state = next_state

            rew += reward[0]
            if np.all(done):
                if np.all(done == 1):
                    t_per_episode.append(env.t)
                    avg_time += env.t
                    is_collision.append(0)
                    acc_num += 1
                elif np.any(done == 3):
                    t_per_episode.append(2000)
                    is_collision.append(1)
                else:
                    t_per_episode.append(1000)
                    is_collision.append(0)
                break
        if if_render and num_episode < 3:
            images = []
            for filename in fig_paths:
                images.append(imageio.imread(filename))
            os.makedirs(os.path.join(val_save_path, "renders/gifs/"), exist_ok=True)
            gif_path = os.path.join(val_save_path, "renders/gifs/", str(num_episode) + ".gif")
            imageio.mimsave(gif_path, images)
        if acc_num == 0:
            avg_len = 0
        else:
            avg_len = avg_time / acc_num
        count = num_episode + 1
        message = "{}/{} --- Evaluation Finished with acc rate {}%: collision rate:{}% Avg Ep Rew: {} Avg time len: {} Duration: {}......".format(
            count, eval_nums,
            acc_num / (count) * 100,
            np.sum(
                is_collision) / (count) * 100,
            rew / (count),
            avg_len,
            round(time.time() - start_time, 2)
        )
        if print_info and count % 10 == 0:
            print(message)

    if if_save:
        np.savetxt(os.path.join(val_save_path, f'time_{name}.txt'),
                   t_per_episode)
        np.savetxt(os.path.join(val_save_path, f'collision_{name}.txt'),
                   is_collision)
    print(message)
    return acc_num, np.sum(is_collision), rew / eval_nums, message


if __name__ == '__main__':
    from net.net_d3qn import ActorCriticD3QN
    from net.net_dqn import ActorCriticDQN
    from agent.agent_d3qn import AgentD3QN
    from agent.agent_dqn import AgentDQN

    from agent.zsc_agent import PursuerAgent

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(description='hyperparameters.')

    parser.add_argument('--model_dir_path', type=str, default=None, help='Path to the model directory')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
    parser.add_argument('-n', '--eval_nums', type=int, default=50, help='Number of evaluations')
    parser.add_argument('--num_agent', type=int, default=3, help='Number of pursuit')
    parser.add_argument('-s', '--set_teammate_methods', type=str, default='selfplay', help='Set teammate methods')
    parser.add_argument('--algo', type=str, default='DQN', help='Algorithm')
    parser.add_argument('--if_render', action='store_true', help='Whether to render')
    parser.add_argument('--if_save', action='store_true', help='Whether to save')

    args = parser.parse_args()

    # model_dir_path = "/home/liyang/ZSC_pursuit/results/selfplay/2024-03-17-05_14_10/model/3400.pt"
    # save_path = "/home/liyang/ZSC_pursuit/results/selfplay/2024-03-17-05_14_10/eval_res/"
    model_dir_path = "results/selfplay-DQN-ego/2024-04-24-08_58_40/best_model/best.pt"
    save_path = "results/selfplay-DQN-ego/2024-04-24-08_58_40/eval_res"
    # model_dir_path = 'results/selfplay-DQN/2024-04-08-06_03_40/model'
    # save_path = "results/selfplay-DQN/2024-04-08-06_03_40/eval_res/"

    eval_nums = args.eval_nums
    set_teammate_methods = args.set_teammate_methods
    algo = args.algo
    if_render = False
    if_save = True
    if algo == "DQN":
        self_net = ActorCriticDQN(num_action=24)  # instantiate the network
    else:
        self_net = ActorCriticD3QN(num_action=24)  # instantiate the network

    config = {
        "num_action": 24,
        "num_state": 4 + 1 + (args.num_agent - 1) * 2,
        "num_p": args.num_agent,
        "num_e": 1,
        "mode": "Valid",
        "num_p": 3,
        "r_velocity": 300,
        "delta_t": 0.1,
        "r_perception": 2000,
        "algorithm": "DQN",
        "episode_max": 4000,
        "batch_size": 128,
        "epsilon_origin": 1,
        "epsilon_decrement": 0.0005,
        "gamma": 0.99,
        "target_replace_iter": 200,
        "memory_size": 1000000,
        "save_path": "./results",
        "method": "selfplay",
        "learning_rate": 3e-5,
        "eval_save_steps": 200,
        "eval_nums": 100
    }
    env = AsyEnvironment(config)

    if model_dir_path[-3:] == ".pt":
        name = model_dir_path.split("/")[-1] + f"_{set_teammate_methods}"
        load_path = model_dir_path
        self_net.load_state_dict(torch.load(load_path))
        print(f"load from {load_path}")
        if algo == "DQN":
            valid_agent = AgentDQN(config_args=None, idx=0, pretrain_net=self_net)
        else:
            valid_agent = AgentD3QN(config_args=None, idx=0, pretrain_net=self_net)
        os.makedirs(save_path, exist_ok=True)
        # env_config, valid_agent, teammate_list, eval_nums, val_save_path, if_save=False, if_render=False
        suc_num, collision_num, t_per_episode, is_collision = validate(config, valid_agent=valid_agent,
                                                                       teammate_list=[valid_agent] * 2,
                                                                       eval_nums=eval_nums,
                                                                       if_render=if_render, val_save_path=save_path,
                                                                       if_save=if_save)
        message = "Evaluation Finished with acc rate {}%: collision rate:{}%......\n".format(suc_num / eval_nums * 100,
                                                                                             collision_num / eval_nums * 100)
        message += f"The ckpt is loaded from {load_path}"

        if if_save:
            os.makedirs(os.path.join(save_path, "renders/res/"), exist_ok=True)
            np.savetxt(os.path.join(save_path, "renders/res/", f'time_{str(name)}.txt'), t_per_episode)
            np.savetxt(os.path.join(save_path, "renders/res/", f'collision_{str(name)}.txt'), is_collision)
            with open(os.path.join(save_path, "renders/res/", f'result_{str(name)}.txt'), "w") as text_file:
                text_file.write(message)
    else:
        for name in range(4000, 3000, -200):
            # for name in range(24800, 23000, -200):
            load_path = os.path.join(model_dir_path, str(name) + ".pt")
            # load_path = os.path.join(model_dir_path, str(name), "0.pt")
            self_net.load_state_dict(torch.load(load_path))
            print(f"load from {load_path}")

            if algo == "DQN":
                valid_agent = AgentDQN(config_args=None, idx=0, pretrain_net=self_net)
            else:
                valid_agent = AgentD3QN(config_args=None, idx=0, pretrain_net=self_net)
            os.makedirs(save_path, exist_ok=True)
            suc_num, collision_num, t_per_episode, is_collision = validate(config, valid_agent=valid_agent,
                                                                           teammate_list=[valid_agent] * 2,
                                                                           eval_nums=eval_nums,
                                                                           if_render=if_render, val_save_path=save_path,
                                                                           if_save=if_save)
            message = "Evaluation Finished with acc rate {}%: collision rate:{}%......\n".format(
                suc_num / eval_nums * 100,
                collision_num / eval_nums * 100)
            message += f"The ckpt is loaded from {load_path}"

            if if_save:
                os.makedirs(os.path.join(save_path, "renders/res/"), exist_ok=True)
                np.savetxt(os.path.join(save_path, "renders/res/", f'time_{str(name)}_{set_teammate_methods}.txt'),
                           t_per_episode)
                np.savetxt(os.path.join(save_path, "renders/res/", f'collision_{str(name)}_{set_teammate_methods}.txt'),
                           is_collision)
                with open(os.path.join(save_path, "renders/res/", f'result_{str(name)}_{set_teammate_methods}.txt'),
                          "w") as text_file:
                    text_file.write(message)
