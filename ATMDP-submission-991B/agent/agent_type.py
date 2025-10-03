from enum import Enum


class AgentType(Enum):
    """
    Enum class for agent types
    """
    PPO = 1
    DQN = 2
    D3QN = 3
    DACOOP = 4
    MAPPO = 5
    VICSEK = 6
    GREEDY = 7
    IPPO = 8
    POAM = 9

    def __str__(self):
        return self.name

    def encrypt_action(self, original_action, static_encrypt_len):
        return self.value * static_encrypt_len + original_action

    @staticmethod
    def get_agent_type_from_value(value):
        for agent_type in AgentType:
            if agent_type.value == value:
                return agent_type
        return None

    @staticmethod
    def get_agent_type_from_string(agent_type_str):
        for agent_type in AgentType:
            if agent_type.name == agent_type_str:
                return agent_type
        return None
