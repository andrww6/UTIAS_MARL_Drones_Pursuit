import numpy as np
import copy

from .zsc_agent import PursuerAgent


class AgentGreedy(PursuerAgent):

    def __init__(self, idx):
        super().__init__(eval_net=None, idx=idx)

    @staticmethod
    def greedy_chaser_velocity(state, agent_orientation, num_agents, num_evaders, captures):
        """
        Choose an action according to epsilon-greedy method.
        Input:
            state: the observation of pursuers
            epsilon: the value of current epsilon
        Output:
            action: the action index chosen, range from 0 to num_action-1, each stands for 360/num_action degree
        """
        state_tem = copy.deepcopy(state).reshape((1, -1))
        # get nearest obstacle distance and orientation
        nearest_obstacle_distance = state_tem[0, num_evaders * 2]
        nearest_obstacle_orientation = state_tem[0, num_evaders * 2 + 1]
        # get target and orientation
        shortest_distance = 100000
        target_orientation = np.zeros((2, 1))

        other_obstacle_distance = []
        other_obstacle_orientation = []
        for i in range(0, num_evaders):
            if (captures[0, i] == 1):
                other_obstacle_distance.append(state_tem[0, i * 2])
                other_obstacle_orientation.append(state_tem[0, i * 2 + 1])
                continue
            if state_tem[0, i * 2] < shortest_distance:
                shortest_distance = state_tem[0, i * 2]
                target_orientation = state_tem[0, i * 2 + 1]
        # get orientation of other agents
        evader_state_num = num_evaders * 2 + 2
        for i in range(0, num_agents - 1):
            other_obstacle_distance.append(state_tem[0, i * 2 + evader_state_num])
            other_obstacle_orientation.append(state_tem[0, i * 2 + 1 + evader_state_num])

        # find the nearest agent and its orientation
        nearest_agent_distance = min(other_obstacle_distance)
        nearest_agent_orientation = other_obstacle_orientation[other_obstacle_distance.index(nearest_agent_distance)]

        if nearest_agent_distance < nearest_obstacle_distance:
            min_distance = nearest_agent_distance
            min_orientation = nearest_agent_orientation
        else:
            min_distance = nearest_obstacle_distance
            min_orientation = nearest_obstacle_orientation

        current_angle_radians = np.arctan2(agent_orientation[1, 0], agent_orientation[0, 0])
        avoidance_threshold = 300

        # Adjust direction if an obstacle is detected within the avoidance threshold
        if min_distance * 5000 < avoidance_threshold:
            # Simple avoidance strategy: turn right or left depending on the obstacle direction
            if min_orientation < 0:
                # Turn right
                change_radians = np.pi / 9
            else:
                # Turn left
                change_radians = - np.pi / 9
            new_angle_radians = current_angle_radians + change_radians
        else:
            # No need to adjust direction
            new_angle_radians = current_angle_radians + target_orientation

        final_degree = np.degrees(new_angle_radians) % 360

        final_move_radians = final_degree * np.pi / 180
        orientation_vector = np.array([np.cos(final_move_radians), np.sin(final_move_radians)])
        return copy.deepcopy(orientation_vector).reshape((2, 1))

    def choose_action(self, **kwargs):
        return 0
