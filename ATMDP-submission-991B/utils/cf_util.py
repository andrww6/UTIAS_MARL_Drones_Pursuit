import json
import torch
import copy
from net.net_d3qn import ActorCriticD3QN
from net.net_dqn import ActorCriticDQN
from net.net_ppo import ActorCriticPPO

from agent.agent_d3qn import AgentD3QN
from agent.agent_dqn import AgentDQN
from agent.agent_ppo import AgentPPO
from agent.agent_dacoop import AgentDACOOP
from agent.agent_vicsek import AgentVicsek
from agent.agent_greedy import AgentGreedy


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def rearrange_config(parser_args):
    print("current train_config path: ", parser_args["config_path"])
    config = load_config(parser_args["config_path"])
    final_config = copy.deepcopy(config)
    for key, value in config.items():
        if key in parser_args.keys():
            if parser_args[key] is not None:
                final_config[key] = parser_args[key]

    for key, value in parser_args.items():
        if key not in final_config.keys():
            final_config[key] = parser_args[key]

    print("current config: ", final_config)
    print("------------------------------------")
    return final_config

def check_duplicates(*lists):
    combined_list = []
    for lst in lists:
        combined_list.extend(lst)

    seen_elements = set()
    duplicates = set()

    for item in combined_list:
        if item in seen_elements:
            duplicates.add(item)
        else:
            seen_elements.add(item)

    if duplicates:
        print("exist duplicates:", duplicates)
        return True, duplicates
    else:
        return False, None
