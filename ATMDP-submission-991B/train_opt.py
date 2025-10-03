import os
import datetime
import random
import time

import wandb
import argparse
import numpy as np
import torch
import time
import math,statistics
import json

from agent.agent_greedy import AgentGreedy
from agent.agent_vicsek import AgentVicsek
from agent.agent_dqn import AgentDQN
from agent.agent_d3qn import AgentD3QN
from agent.agent_dacoop import AgentDACOOP
from agent.agent_ppo import AgentPPO
from validation.val import validate_s_env

from env.environment import environment
from utils.cf_util import rearrange_config
from utils.run_util import set_seed, RunningStat, make_env
from env.parall import ParallelEnv
from utils.agent_util import get_agent_type, create_agent, random_teammates
from validation.val import validate_p_env as validate_env
from hygraph.hypergraph import Preference_Graph, Hypergraph
from itertools import combinations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hyperparameters.')
    # default set as DQN config path
    parser.add_argument('--config_path', type=str, default="./config/opt_config.json",
                        help='the path to agent config')
    parser.add_argument('--seed', type=int, default=1, help='Value for seed')
    # env parameters
    parser.add_argument('--mode', type=str, default='Train', help='the mode of environment')
    parser.add_argument('--num_p', type=int, default=4, help='the number of pursuit')
    parser.add_argument('--num_e', type=int, default=2, help='the number of evader')
    parser.add_argument('--num_action', type=int, default=1, help='the number environment action')
    parser.add_argument('--delta_t', type=float, default=0.1, help='time interval (s)')
    parser.add_argument('-rv', '--r_velocity', type=int, default=300, help='velocity of pursuers (mm/s)')
    parser.add_argument('-ev', '--e_velocity', type=int, default=600, help='velocity of evaders (mm/s)')
    parser.add_argument('--r_perception', type=int, default=2000, help='sense range of pursuers')
    parser.add_argument('--obstacle_num', type=int, default=5, help='the number of obstacles')
    parser.add_argument('--ctrl_num', type=int, default=2, help='the number of train agent controls player')

    # train parameters
    parser.add_argument('--neg_reward', type=int, default=-10, help='reward when collision')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--env_num', type=int, default=4, help='the number of env')
    parser.add_argument('--env_idx', type=int, default=None, help='env index id')
    parser.add_argument('--g_sim_num', type=int, default=10, help='the number of simulation to build model')
    
    parser.add_argument('--epoches', type=int, default=200, help='train epochs')
    parser.add_argument('--in_epoches', type=int, default=5, help='train times for each random agent')
    parser.add_argument('--eval_save_steps', type=int, default=5000, help='eval epoches')
    parser.add_argument('-p', '--population_size', type=int, default=5, help='train agent population size')
    parser.add_argument('--more_r', action='store_true', help='disable additional reward mechanism to avoid collision')
    parser.add_argument('--old_env', action='store_false', help='enable the use of the old environment')

    # the parameters are about wandb
    parser.add_argument('--wandb', action='store_true', help='if we use wandb')
    parser.add_argument('--proj', type=str, default='MDPAT', help='Value for wandb project name')

    args = parser.parse_args()
    args = vars(args)
    set_seed(args['seed'])

    config = rearrange_config(args)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # save path
    save_path = config["res_path"]
    save_path = os.path.join(save_path, str(args["seed"]), current_time)
    os.makedirs(save_path, exist_ok=True)
    config["save_path"] = save_path
    print(f"save path is: {save_path}")
    
    if config["wandb"]:
        run = wandb.init(
            project=config["proj"],
            name="{}(ctrl:{})VS{}-seed-({})-time-({})".format(config["num_p"], config["ctrl_num"], config["num_e"], config["seed"], current_time),
            group="OPT-shared",
            config=config,
            sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
            save_code=True,
            dir=config["res_path"],
            entity="MDPAT"
        )

    # create environment
    # train_env = AsyEnvironment(config)
    env_num = config["env_num"]
    env_instances = [make_env(config) for _ in range(env_num)]
    parallel_env = ParallelEnv(env_instances)

    train_env = environment(config)
    config["action_dim"] = train_env.action_dim
    num_agent = config["num_p"]
    ctrl_num = config["ctrl_num"]
    unctrl_num = num_agent - ctrl_num
    population_size = config["population_size"]

    # running stat
    running_stat = RunningStat()
    device = torch.device(config['device'])
    
    # create empty population
    population = []
    # create the learner agent
    train_agent = create_agent(config, 666)
    train_agent.eval_net.to(device)

    # Hyper Parameters
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    training_step = 0
    best_succ_acc = -1
    start_t = time.time()
    train_index = train_agent.idx
    hy_graph = Hypergraph(r=unctrl_num+1)
    
    print("Now, building the initial population......")
    edge=""
    for i in range(unctrl_num+1):
        tmp_agent = create_agent(config, i)
        tmp_agent.eval_net.load_state_dict(train_agent.eval_net.state_dict())
        tmp_agent.eval_net.to(device)
        population.append(tmp_agent)
        edge+=str(i)
    
    print("the initial population built... now evaluate it and build the inital hypergraph...")
    val_suc_num, val_collision_num, ep_rew, message = validate_env(env_config=config,
                                                                    valid_agent=population[-1],
                                                                    teammate_list=population[:unctrl_num],
                                                                    eval_nums=config["g_sim_num"],
                                                                    val_save_path=None,
                                                                    print_info=False,
                                                                    device=device)
    
    hy_graph.add_edge(edge, weight=sum(ep_rew))
    hy_graph.print_info()
    
    # for all epoches
    # for ep in range(config["epoches"]):
    ep = 0
    flag = 0
    while training_step < config["training_step_max"]:
        ep+=1
        recoder = {
            "loss_actor": [],
            "loss_critic":[],
            "loss_entropy":[],
            "std":[],
            "episode_return":[]
        }
        
        now_PG = Preference_Graph(hy_graph)
        # for each in epoch the population is fixed.
        in_epoches = 1 #min(config["in_epoches"], math.comb(len(population), unctrl_num)) because line 226
        for in_ep in range(in_epoches):
            train_agent.eval_net.train()
            # sample teammates 
            teammate_list = [population[int(i)] for i in now_PG.sample(num=unctrl_num)]
            agent_type = get_agent_type(teammate_list)

            episode_return_list = np.zeros((1, env_num))
            last_done_array = np.array([np.zeros((1, num_agent)) for _ in range(env_num)])

            state_array,_, _ = parallel_env.reset()

            while train_agent.need_more_data():
                # reset the environments vector
                chooses_actions_array = np.zeros((env_num, num_agent))  # Assuming action space dimensionality fits
                agent_type_array = [agent_type] * env_num

                current_env_state = state_array
                for i in range(num_agent):
                    if i < ctrl_num:
                        current_agent = train_agent
                    else:
                        current_agent = teammate_list[i - ctrl_num]
                    state_temp = current_env_state[:, :, i:i + 1]
                    action_temp = current_agent.choose_action(state_temp, num_agent, device=device)
                    action_temp = action_temp[:, 0]
                    # print(action_temp.shape)
                    chooses_actions_array[:, i] = action_temp

                # Execute actions in all environments and receive new states and rewards
                next_state_array,_, rewards, dones, _ = parallel_env.step(chooses_actions_array, agent_type_array, ctrl_num)
                # print(next_state_array.shape)
                for env_index in range(env_num):
                    last_done = last_done_array[env_index]
                    current_done = dones[env_index]
                    for i in range(ctrl_num):# or all agents
                        if not np.ravel(last_done)[i]:
                            episode_return_list[0, env_index] += rewards[env_index, 0, i]
                            train_agent.store_transition(state_array[env_index][:, i:i + 1],
                                                        chooses_actions_array[env_index, i:i + 1],
                                                        rewards[env_index, :, i:i + 1],
                                                        next_state_array[env_index, :, i:i + 1],
                                                        current_done[:, i:i + 1], 0)
                    if np.all(current_done):
                        state_inv,_, _ = parallel_env.reset_ind(env_index)
                        last_done_inv = np.zeros((1, num_agent))
                        state_array[env_index] = state_inv
                        last_done_array[env_index] = last_done_inv
                    else:
                        state_array[env_index] = next_state_array[env_index]
                        last_done_array[env_index] = current_done

            loss_actor, loss_critic, loss_entropy, training_step = train_agent.learn(train_env.num_state,
                                                                                    train_env.action_dim,
                                                                                    num_agent,
                                                                                    device, training_step)
            recoder["loss_actor"].append(loss_actor)
            recoder["loss_critic"].append(loss_critic)
            recoder["loss_entropy"].append(loss_entropy)
            recoder["std"].append(torch.exp(train_agent.eval_net.logstd).detach().to('cpu').numpy().ravel(),)
            episode_return = np.mean(episode_return_list)
            recoder["episode_return"].append(episode_return)
        if ep < population_size * 2:
            pass # if the iter number is too small, just update the population,then train
        else:
            if training_step - flag < config["eval_save_steps"]:
                continue
            flag = training_step
        print("Now update the hypergraph.....")
        if len(population) < population_size:
            tmp_agent = create_agent(config, len(population))
            tmp_agent.eval_net.to(device)
            tmp_agent.eval_net.load_state_dict(train_agent.eval_net.state_dict())
            population.append(tmp_agent)
            train_where = len(population) - 1
            print("Updated population by adding {}".format(len(population)))
        else:
            del_node = now_PG.min_eta()
            hy_graph.del_node(str(del_node)) # delelt the elimislated node
            train_where = int(del_node)

            tmp_agent = create_agent(config, idx=train_where)
            tmp_agent.eval_net.to(device)
            tmp_agent.eval_net.load_state_dict(train_agent.eval_net.state_dict())
            population[train_where] = tmp_agent
            print("Updated population by replacing {}".format(train_where))
            
        # Create a new list excluding the item at index train_where
        filtered_population = [item for idx, item in enumerate(population) if idx != train_where]
        sample_num = min(20, math.comb(len(filtered_population), unctrl_num))
        combinations_list = list(combinations(filtered_population, unctrl_num))
        sampled_teamates = random.sample(combinations_list, sample_num)
        # Sample from the filtered population
        for teammate_list in sampled_teamates:
            edge = f"{train_where}"
            for agent in teammate_list:
                edge += str(agent.idx)
                    
            val_suc_num, val_collision_num, ep_rew, message = validate_env(env_config=config,
                                                                            valid_agent=train_agent,
                                                                            teammate_list=teammate_list,
                                                                            eval_nums=config["g_sim_num"],
                                                                            val_save_path=None,
                                                                            print_info=False,
                                                                            device=device)
        
        
            hy_graph.add_edge(edge, weight=np.mean(ep_rew))
        hy_graph.print_info()
        
        lr_now = lr * (1 - training_step / config["training_step_max"])
        if lr_now < 1e-5:
            lr_now=1e-5
        for g in train_agent.optimizer.param_groups:
            g['lr'] = lr_now

        log_data = {
                "train/loss_actor": np.mean(recoder["loss_actor"]),
                "train/loss_critic": np.mean(recoder["loss_critic"]),
                "train/loss_entropy": np.mean(recoder["loss_entropy"]),
                "train/std": np.mean(recoder["std"]),
                "train/training_step": np.mean(training_step),
                "train/episode_return": np.mean(recoder["episode_return"]),
                "train/lr": lr_now,
            }
        if config["wandb"]:
            wandb.log(log_data)
        print(f"----> epoch {ep}")
        print(log_data)

        if ep < population_size * 2:
            pass # if the iter number is too small, just update the population,then train
        else:
            # save
            model_path = os.path.join(save_path, 'model')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_save_path = os.path.join(model_path, str(ep))
            torch.save(train_agent.eval_net.state_dict(), model_save_path + '.pt')
            # set eval mode
            
            now_pg = Preference_Graph(hy_graph)
            median_node, max_node = now_pg.eval_teammates()
            
            print(f"The nodes in median and max is {median_node}, {max_node}")
            train_agent.eval_net.eval()
            # the uncontrolled is 2!!!
            if unctrl_num == 2:
                teammate_list = [population[median_node], population[max_node]]
            elif unctrl_num == 1:
                teammate_list = [population[max_node]]
            val_suc_rate, val_collision_rate, t_per_episode, is_collision = validate_env(env_config=config,
                                                                                            valid_agent=train_agent,
                                                                                            teammate_list=teammate_list,
                                                                                            eval_nums=config["eval_nums"],
                                                                                            val_save_path=None,
                                                                                            device=device)
            
            
            # set train mod
            train_agent.eval_net.train()
            

            log_data = {
                "val/val_suc_rate": val_suc_rate,
                "val/val_collision_rate": val_collision_rate
            }
            if val_suc_rate > best_succ_acc:
                best_succ_acc = val_suc_rate
                best_path = os.path.join(save_path, 'best')
                os.makedirs(best_path, exist_ok=True)
                torch.save(train_agent.eval_net.state_dict(), os.path.join(best_path, "best.pt"))
                message = "New Best Model: \n\t ep:{}; training_step:{}; suc_rate:{}, collision_rate:{}".format(
                    ep, training_step, round(episode_return, 2), val_suc_rate, val_collision_rate)
                
                converted_data = {key: int(value) if isinstance(value, np.integer) else value for key, value in log_data.items()}
                with open(os.path.join(best_path, "best_info.json"), 'w') as json_file:
                    json.dump(converted_data, json_file, indent=4)  # indent=4 用于美化格式
            if config["wandb"]:
                wandb.log(log_data)
            print(log_data)
        start_t = time.time()

    parallel_env.close()