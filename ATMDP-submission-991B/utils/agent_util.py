import copy
import torch
import wandb
import random
import itertools

from agent.agent_type import AgentType
from agent.agent_greedy import AgentGreedy
from agent.agent_vicsek import AgentVicsek
from agent.agent_dqn import AgentDQN
from agent.agent_d3qn import AgentD3QN
from agent.agent_dacoop import AgentDACOOP
from agent.agent_ppo import AgentPPO
from agent.agent_mappo import AgentMAPPO
from agent.agent_ippo import AgentIPPO
from agent.agent_poam import AgentPOAM
from common.constants import STATIC_ENCRYPT_LEN

from net.net_dqn import ActorCriticDQN
from net.net_d3qn import ActorCriticD3QN
from net.net_ppo import ActorCriticPPO
from net.net_mappo import ActorCriticMAPPO
from net.net_dacoop import ActorCriticDACOOP


def get_agent_type(teammate_list):
    agent_types = []
    for agent in teammate_list:
        agent_types.append(get_single_agent_type(agent))
    return agent_types


def get_single_agent_type(agent):
    if isinstance(agent, AgentDQN):
        return AgentType.DQN
    elif isinstance(agent, AgentD3QN):
        return AgentType.D3QN
    elif isinstance(agent, AgentDACOOP):
        return AgentType.DACOOP
    elif isinstance(agent, AgentPPO):
        return AgentType.PPO
    elif isinstance(agent, AgentMAPPO):
        return AgentType.MAPPO
    elif isinstance(agent, AgentPOAM):
        return AgentType.POAM
    elif isinstance(agent, AgentIPPO):
        return AgentType.IPPO
    elif isinstance(agent, AgentVicsek):
        return AgentType.VICSEK
    elif isinstance(agent, AgentGreedy):
        return AgentType.GREEDY
    else:
        print(agent)
        raise ValueError("Unknown agent type")


def joint_action(actions, agent_types):
    joint_actions = copy.deepcopy(actions)
    for i in range(0, len(agent_types)):
        joint_actions[:, i + 1] = agent_types[i].encrypt_action(actions[:, i + 1], STATIC_ENCRYPT_LEN)
    return joint_actions


def create_agent(config_args, idx, wandb_logger=False, ctrl_num=1):
    agent = None
    if config_args["algorithm"] == "DQN":
        agent = AgentDQN(config_args=config_args, idx=idx, ctrl_num=ctrl_num)
    elif config_args["algorithm"] == "D3QN":
        agent = AgentD3QN(config_args=config_args, idx=idx, ctrl_num=ctrl_num)
    elif config_args["algorithm"] == "DACOOP":
        agent = AgentDACOOP(config_args=config_args, idx=idx, ctrl_num=ctrl_num)
    elif config_args["algorithm"] == "PPO":
        agent = AgentPPO(config_args=config_args, idx=idx, ctrl_num=ctrl_num)
    elif config_args["algorithm"] == "MAPPO":
        agent = AgentMAPPO(config_args=config_args, idx=idx, ctrl_num=ctrl_num)
    elif config_args["algorithm"] == "IPPO":
        agent = AgentIPPO(config_args=config_args, idx=idx, ctrl_num=ctrl_num)
    elif config_args["algorithm"] == "POAM":
        agent = AgentPOAM(config_args=config_args, idx=idx, ctrl_num=ctrl_num)
    elif config_args["algorithm"] == "VICSEK":
        agent = AgentVicsek(idx=idx)
        return agent
    elif config_args["algorithm"] == "GREEDY":
        agent = AgentGreedy(idx=idx)
        return agent
    agent_name = agent.name
    if wandb_logger:
        print("need to fill what information you want to log")
        #     wandb.define_metric(f"{agent_name}_step")
        #     wandb.define_metric(f"time/iterations/{agent_name}", step_metric=f"{agent_name}_step")
        #     wandb.define_metric(f"time/episodes/{agent_name}", step_metric=f"{agent_name}_step")
        #     wandb.define_metric(f"time/fps/{agent_name}", step_metric=f"{agent_name}_step")
        #     wandb.define_metric(f"time/time_elapsed/{agent_name}", step_metric=f"{agent_name}_step")
        #     wandb.define_metric(f"time/total_timesteps/{agent_name}", step_metric=f"{agent_name}_step")
        #     wandb.define_metric(f"rollout/ep_rew_mean/{agent_name}", step_metric=f"{agent_name}_step")
        #     wandb.define_metric(f"rollout/ep_len_mean/{agent_name}", step_metric=f"{agent_name}_step")
        #     wandb.define_metric(f"rollout/success_rate/{agent_name}", step_metric=f"{agent_name}_step")
        wandb.define_metric(f"{agent_name}_iter")
        wandb.define_metric(f"train/loss_actor/{agent_name}", step_metric=f"{agent_name}_iter")
        wandb.define_metric(f"train/loss_critic/{agent_name}", step_metric=f"{agent_name}_iter")
        wandb.define_metric(f"train/loss_entropy/{agent_name}", step_metric=f"{agent_name}_iter")
        wandb.define_metric(f"train/std/{agent_name}", step_metric=f"{agent_name}_iter")
        wandb.define_metric(f"train/training_step/{agent_name}", step_metric=f"{agent_name}_iter")
        wandb.define_metric(f"train/episode_return/{agent_name}", step_metric=f"{agent_name}_iter")
        wandb.define_metric(f"train/fps/{agent_name}", step_metric=f"{agent_name}_iter")

        wandb.define_metric(f"train/loss/{agent_name}", step_metric=f"{agent_name}_iter")

        wandb.define_metric("eval_iter")
        wandb.define_metric(f"eval/succ_rate/{agent_name}", step_metric="eval_iter")
        wandb.define_metric(f"eval/coll_rate/{agent_name}", step_metric="eval_iter")
        wandb.define_metric(f"eval/episode_return/{agent_name}", step_metric="eval_iter")
    if agent is None:
        raise ValueError("Unknown agent type")
    return agent


def random_teammates(population, config_args, out_index):
    set_method = config_args["method"]
    if set_method == 'sp':
        agent_size = int(config_args["num_p"] / population[0].ctrl_num)
        return [population[0]] * (agent_size - 1)
    elif set_method == 'pbt':
        remaining_indices = [i for i in range(len(population)) if i != out_index]

        remaining_control_num = config_args["num_p"] - population[out_index].ctrl_num
        # 这里应该有个agent_size，也就是除了num_p(追逐者/无人机数量之外，如果有限制的话，应该还要加一个max_agent_size的属性)
        all_valid_combos = __find_all_combos_fixed_sum__(population, remaining_indices,
                                                         config_args["num_p"] - 1,
                                                         remaining_control_num)
        assert all_valid_combos, "There is not valid combo for controller agent"
        selected_indices = random.choice(all_valid_combos)
        return [population[i] for i in selected_indices]


def eval_random_teammates(valid_agent, teammate_candidates, target, eval_method):
    res = []
    if eval_method == 'fcp':
        valid_agent_temp = copy.deepcopy(valid_agent)
        valid_agent_temp.ctrl_num = 1
        res = [valid_agent_temp] * target
    elif eval_method == 'zsc':
        # 这里应该有个agent_size，也就是除了num_p(追逐者/无人机数量之外，如果有限制的话，应该还要加一个max_agent_size的属性)
        all_valid_combos = __find_all_combos_fixed_sum__(teammate_candidates, target)
        assert len(all_valid_combos) != 0, "There is not valid combo for controller agent"
        res = random.choice(all_valid_combos)
    else:
        # 如果没有设定eval_method, 那么默认teammates_candidates就是输入的teammates
        res = teammate_candidates
    total_ctrl = sum(candidate.ctrl_num for candidate in res)
    assert total_ctrl == target, "direct teammates does not match the condition"
    return res


def __find_all_combos_fixed_sum__(teammate_candidates, desired_sum):
    """
    在 teammate_candidates 中，枚举所有可能的对象组合，
    并筛选出这些组合中对象 ctrl_num 的总和恰好等于 desired_sum 的组合。

    参数:
        teammate_candidates (list): 候选者对象列表，每个对象具有属性 ctrl_num。
        desired_sum (int): 目标总和。

    返回:
        list: 满足条件的组合列表，每个元素是一个元组，表示一个组合的对象。
              如果没有任何组合满足条件，则返回空列表 []。
    """
    if not teammate_candidates or desired_sum < 0:
        return []  # 边界条件检查，返回空列表

    all_valid_combos = []

    # 遍历组合长度从 1 到 len(teammate_candidates)
    for r in range(1, len(teammate_candidates) + 1):
        for combo in itertools.combinations(teammate_candidates, r):
            # 计算组合中对象的 ctrl_num 总和
            total_ctrl = sum(candidate.ctrl_num for candidate in combo)
            if total_ctrl == desired_sum:
                all_valid_combos.append(combo)

    return all_valid_combos


def dic_to_agent(pre_trained_net_pth_dict, config, num_action=24, action_dim=1, start_idx=0):
    pre_train_teammate_list = []
    # load pre-trained agents
    if pre_trained_net_pth_dict is not None:
        for key in pre_trained_net_pth_dict.keys():
            file_path_list = pre_trained_net_pth_dict[key]
            for file in file_path_list:
                teammate_temp = None
                if key in ['D3QN']:
                    net_temp = ActorCriticD3QN(num_action)
                    net_temp.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
                    teammate_temp = AgentD3QN(config_args=config, idx=len(pre_train_teammate_list) + 1 + start_idx,
                                              pretrain_net=net_temp)
                elif key in ['DQN']:
                    net_temp = ActorCriticDQN(num_action)
                    net_temp.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
                    teammate_temp = AgentDQN(config_args=config, idx=len(pre_train_teammate_list) + 1 + start_idx,
                                             pretrain_net=net_temp)
                elif key in ['PPO']:
                    net_temp = ActorCriticPPO(action_dim)
                    net_temp.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
                    teammate_temp = AgentPPO(config_args=config, idx=len(pre_train_teammate_list) + 1 + start_idx,
                                             pretrain_net=net_temp)
                elif key in ['MAPPO']:
                    net_temp = ActorCriticMAPPO(action_dim)
                    net_temp.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
                    teammate_temp = AgentMAPPO(config_args=config, idx=len(pre_train_teammate_list) + 1 + start_idx,
                                               pretrain_net=net_temp)
                elif key in ['VICSEK']:
                    teammate_temp = AgentVicsek(idx=len(pre_train_teammate_list) + 1 + start_idx)
                elif key in ['GREEDY']:
                    teammate_temp = AgentGreedy(idx=len(pre_train_teammate_list) + 1 + start_idx)
                else:
                    raise ValueError(
                        "Error Message: key should be in ['D3QN', 'DQN', 'PPO', 'VICSEK', 'GREEDY'] in pre_trained_net_dict. current key is: ",
                        key)

                pre_train_teammate_list.append(teammate_temp)
        print('agents transfer successfully. teammate list size:', len(pre_train_teammate_list))
        return pre_train_teammate_list
    else:
        raise ValueError("Error Message: pre_trained_net_dict should not be None or empty when teammate method  when "
                         "teammate method is pretrain")
