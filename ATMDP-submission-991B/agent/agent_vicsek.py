import numpy as np
import copy
import env.APF_function_for_DQN as APF_function_for_DQN
# from env.environment import Environment

from agent.zsc_agent import PursuerAgent


class AgentVicsek(PursuerAgent):

    def __init__(self, idx):
        super().__init__(eval_net=None, idx=idx)

    # The function to compute vicsek velocity command
    # This function is standalone, which means that it can be placed anywhere after removing the input "self"
    @staticmethod
    def vicsek_chaser_velocity(pos_all_chaser_in: np.ndarray,
                               vel_all_chaser_in: np.ndarray,
                               index_chaser: int,
                               pos_evader_in: np.ndarray,
                               vel_evader_in: np.ndarray,
                               pos_obstacle: np.array,
                               v_max_e: float = 8,
                               v_max_c: float = 6,
                               r_wall: float = 0.20,
                               r_cd: float = 0.25,
                               coefficient_f_1: float = 0.1,
                               r_inter: float = 5,
                               coefficient_f_2: float = 0.5,
                               c_inter: float = 0.8,
                               c_soft_wall = 1):
        """
        This function is used to calculate the chaser's displacement.
        Ref: Janosov, M., et al. (2017). "Group chasing tactics: how to catch a faster prey."
        Input:
            pos_all_chaser_in: the positions of all chasers
            vel_all_chaser_in: the velocities of all chasers
            index_chaser: the index of the chaser
            pos_evader_in: the position of the evader
            vel_evader_in: the velocity of the evader
            pos_obstacle: the position of the obstacle
            v_max_e: the maximum speed of the evader
            v_max_c: the maximum speed of the chaser
            r_wall: the radius of the wall
            r_cd: the capture distance
            coefficient_f_1: the coefficient of the chasing force
            r_inter: the interaction radius
            coefficient_f_2: the coefficient of the long-term repulsion
            c_inter: the coefficient of the long-term repulsion
        Output:
            vicsek_v: the velocity of the chaser
        """
        vicsek_v = np.zeros((2, 1))

        # interaction with the wall of the arena: eq.5
        s_function = lambda r_i, r_a, r_wall: \
            -1 / 2 * np.sin(np.pi / r_wall * (r_i - r_a) - np.pi / 2) - 1 / 2 if (r_a <= r_i <= r_a + r_wall) \
                else (-1 if (r_i > r_a + r_wall)
                      else 0)  # eq.6
        # design wall function instead of circle shape s_function
        wall_function = lambda d: -1 / 2 * np.sin(np.pi / r_wall * d - np.pi / 2) - 1 / 2 if (0 < d <= r_wall) else 0
        pos_all_chaser = copy.deepcopy(pos_all_chaser_in) / 1000
        vel_all_chaser = copy.deepcopy(vel_all_chaser_in) / 1000
        pos_evader = copy.deepcopy(pos_evader_in) / 1000
        vel_evader = copy.deepcopy(vel_evader_in) / 1000
        pos_chaser = pos_all_chaser[:, index_chaser:index_chaser + 1]
        vel_chaser = vel_all_chaser[:, index_chaser:index_chaser + 1]
        r_i_vector = pos_chaser - pos_obstacle / 1000
        v_i_vector = vel_chaser
        r_i = np.linalg.norm(r_i_vector)
        v_max_k = np.max([v_max_e, v_max_c])
        soft_wall_term = wall_function(r_i) * (v_max_k * r_i_vector / r_i)

        # collision-avoiding short-term repulsion:
        step_function = lambda x: 1 if x > 0 else 0
        pos_all = np.hstack((pos_all_chaser, pos_obstacle / 1000))  # column vector

        # short_repulsion_term = 0
        # for j in range(pos_all.shape[1]):
        #     if not index_chaser == j:
        #         d_ij = pos_chaser - pos_all[:, j:j + 1]
        #         d_ij_norm = np.linalg.norm(d_ij)
        #         short_repulsion_term += (d_ij_norm - r_cd) / d_ij_norm * d_ij * step_function(r_cd - d_ij_norm)

        d_ij_norm = np.linalg.norm(pos_chaser - pos_all_chaser, axis=0)
        # d_ij_norm = np.sort(d_ij_norm)
        d_index = np.argsort(d_ij_norm)
        min_id = d_index[1]
        d_ij = pos_all_chaser[:, min_id:min_id + 1]
        d_ij_norm = d_ij_norm[min_id]
        short_repulsion_term = (d_ij_norm) / d_ij_norm * d_ij * step_function(r_cd - d_ij_norm)

        if not np.linalg.norm(short_repulsion_term) == 0:
            short_repulsion_term = v_max_k * short_repulsion_term / np.linalg.norm(short_repulsion_term)

        # direct chasing force: eq.10
        r_evader_to_chaser = pos_chaser - pos_evader
        v_evader_to_chaser = vel_chaser - vel_evader
        distance = np.linalg.norm(r_evader_to_chaser)
        if not distance == 0:
            chasing_term = r_evader_to_chaser / distance - coefficient_f_1 * v_evader_to_chaser / distance ** 2
            chasing_term = v_max_c * chasing_term / np.linalg.norm(chasing_term)
        else:
            chasing_term = 0

        # long-term repulsion between chasers:
        long_repulsion_term = 0
        for j in range(pos_all_chaser.shape[1]):
            if not index_chaser == j:
                d_ij = pos_chaser - pos_all_chaser[:, j:j + 1]
                d_ij_norm = np.linalg.norm(d_ij)
                v_ij = vel_chaser - vel_all_chaser[:, j:j + 1]
                long_repulsion_term += d_ij * (
                        d_ij_norm - r_inter) / d_ij_norm - coefficient_f_2 * v_ij / d_ij_norm ** 2
        if not np.linalg.norm(long_repulsion_term) == 0:
            long_repulsion_term = c_inter * v_max_c * long_repulsion_term / np.linalg.norm(long_repulsion_term)

        # sum up: eq.14
        vicsek_v = - c_soft_wall * soft_wall_term - short_repulsion_term - chasing_term - long_repulsion_term
        # vector1 = - 2*soft_wall_term - chasing_term # + soft_wall_term  # calculate F_ar
        # vector2 = - chasing_term
        # vector3 = - short_repulsion_term
        # if np.dot(vector1[:,0], vector2[:,0]) < 0:  # if the angle between F_ar and F_a exceeds 90 degree
        #     # move according to wall following rules
        #     vicsek_v = APF_function_for_DQN.wall_follow(vel_chaser, -2*soft_wall_term, -short_repulsion_term,
        #                                               np.linalg.norm(pos_obstacle - pos_chaser))
        # if np.dot(vector3[:,0], vector2[:,0]) < 0:  # if the angle between F_ar and F_a exceeds 90 degree
        #     vicsek_v = APF_function_for_DQN.wall_follow(vel_chaser, -short_repulsion_term, -chasing_term,
        #                                               np.linalg.norm(pos_all_chaser[:, min_id] - pos_chaser))

        return vicsek_v / np.linalg.norm(vicsek_v)

    def choose_action(self, **kwargs):
        return 0