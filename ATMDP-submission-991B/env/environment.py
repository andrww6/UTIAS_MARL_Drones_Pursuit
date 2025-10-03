import numpy as np
import env.APF_function_for_DQN as APF_function_for_DQN
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from agent.agent_type import AgentType
from agent.agent_escaper import escaper
from agent.agent_vicsek import AgentVicsek
from agent.agent_greedy import AgentGreedy
import copy
import random
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import io


class environment():
    def __init__(self, config):
        # environment features
        self.algorithm = config['algorithm']
        self.num_agent = config['num_p']
        self.ctrl_num = config['ctrl_num']
        self.num_evader = config['num_e']
        self.more_r = config['more_r']
        self.old_env = config['old_env']
        self.task = config['task']
        # if self.algorithm in ['PPO']:
        self.action_dim = 1  # dimension of action space (PPO)
        # elif self.algorithm in ['DQN', 'D3QN']:
        self.num_action = 24  # the number of discretized actions (APF)

        self.num_state = 2 * self.num_evader + 2 + (self.num_agent - 1) * 2  # dimension of state space
        self.d3qn_num_state = 2 * self.num_evader + 2 + 1 + (self.num_agent - 1) * 2  # dimension of state space
        self.t = 0  # timestep
        self.v = config["r_velocity"]  # velocity of pursuers (mm/s)
        self.e_v = config["e_velocity"]
        self.delta_t = config["delta_t"]  # time interval (s)
        self.r_perception = config["r_perception"]  # sense range of pursuers
        # agent features
        self.wall_following = np.zeros((1, self.num_agent))
        self.agent_orientation = np.zeros((2, self.num_agent))
        self.agent_orientation_last = np.zeros((2, self.num_agent))
        self.agent_orientation_origin = np.zeros((2, self.num_agent))
        self.agent_position = np.zeros((2, self.num_agent))
        self.agent_position_last = np.zeros((2, self.num_agent))
        self.agent_position_origin = np.zeros((2, self.num_agent))
        self.obstacle_closest = []
        self.state = np.zeros((self.num_state, self.num_agent))
        self.global_state = np.zeros((self.num_state, self.num_agent))
        # evader features
        self.capture = np.zeros((1, self.num_evader))
        self.target_position = np.zeros((2, self.num_evader))
        self.target_position_last = np.zeros((2, self.num_evader))
        self.target_orientation = np.zeros((2, self.num_evader))
        self.escaper_slip_flag = [0] * self.num_evader
        self.escaper_wall_following = [0] * self.num_evader
        self.escaper_zigzag_flag = [0] * self.num_evader
        self.last_e = np.zeros((2, self.num_evader))
        self.zigzag_count = [0] * self.num_evader
        self.zigzag_last = np.zeros((2, self.num_evader))
        self.neg_reward = config["neg_reward"]
        # obstacle feature
        self.obstacle_num = config["obstacle_num"]
        if self.obstacle_num > 5 or self.obstacle_num <= 0:
            raise ValueError("The obstacle_num of environment must be in the range of 1 to 5.")
        self.env_idx = config["env_idx"]

        if self.env_idx == 0:
            self.boundary = APF_function_for_DQN.generate_boundary(
                shape='rectangle',
                step=51,
                point1=np.array([[0.0], [0]]),
                point2=np.array([[3600], [0]]),
                point3=np.array([[3600], [5000]]),
                point4=np.array([[0], [5000]]))
            # init obstacles matrix
            obstacles = [
                APF_function_for_DQN.generate_boundary(
                    shape='rectangle',
                    step=11,
                    point1=np.array([[1000.0], [2450]]),
                    point2=np.array([[2500], [2450]]),
                    point3=np.array([[2500], [2550]]),
                    point4=np.array([[1000], [2550]]),
                ),

            ]
        elif self.env_idx == 1:
            self.boundary = APF_function_for_DQN.generate_boundary(
                shape='rectangle',
                step=51,
                point1=np.array([[0.0], [0]]),
                point2=np.array([[3600], [0]]),
                point3=np.array([[3600], [5000]]),
                point4=np.array([[0], [5000]]))
            # init obstacles matrix
            obstacles = [
                APF_function_for_DQN.generate_boundary(
                    shape='square',
                    step=10,
                    center=np.array([[1200.0], [1500]]),
                    side_length=400,
                ),
                APF_function_for_DQN.generate_boundary(
                    shape='square',
                    step=10,
                    center=np.array([[1200.0], [3500]]),
                    side_length=400,
                ),
                APF_function_for_DQN.generate_boundary(
                    shape='circle',
                    step=10,
                    center=np.array([[2400.0], [2500]]),
                    radius=200,
                ),
            ]
        elif self.env_idx == 2:
            self.boundary = APF_function_for_DQN.generate_boundary(
                shape='rectangle',
                step=51,
                point1=np.array([[0.0], [0]]),
                point2=np.array([[3600], [0]]),
                point3=np.array([[3600], [5000]]),
                point4=np.array([[0], [5000]]))
            # init obstacles matrix
            obstacles = [
                APF_function_for_DQN.generate_boundary(
                    shape='rectangle',
                    step=11,
                    point1=np.array([[900.0], [1000]]),
                    point2=np.array([[1550], [1000]]),
                    point3=np.array([[1550], [1100]]),
                    point4=np.array([[900], [1100]]),
                ),
                # APF_function_for_DQN.generate_boundary(
                #     np.array([[900.0], [1000]]), np.array([[1550], [1000]]),
                #     np.array([[1550], [1100]]), np.array([[900], [1100]]), 11
                # ),

                APF_function_for_DQN.generate_boundary(
                    shape='rectangle',
                    step=11,
                    point1=np.array([[2050.0], [1000]]),
                    point2=np.array([[2700], [1000]]),
                    point3=np.array([[2700], [1100]]),
                    point4=np.array([[2050], [1100]]),
                ),
                # APF_function_for_DQN.generate_boundary(
                #     np.array([[2050.0], [1000]]), np.array([[2700], [1000]]),
                #     np.array([[2700], [1100]]), np.array([[2050], [1100]]), 11
                # ),

                APF_function_for_DQN.generate_boundary(
                    shape='rectangle',
                    step=11,
                    point1=np.array([[1400.0], [2450]]),
                    point2=np.array([[2200], [2450]]),
                    point3=np.array([[2200], [2550]]),
                    point4=np.array([[1400], [2550]]),
                ),
                # APF_function_for_DQN.generate_boundary(
                #     np.array([[1400.0], [2450]]), np.array([[2200], [2450]]),
                #     np.array([[2200], [2550]]), np.array([[1400], [2550]]), 11
                # ),

                APF_function_for_DQN.generate_boundary(
                    shape='rectangle',
                    step=11,
                    point1=np.array([[900.0], [3900]]),
                    point2=np.array([[1550], [3900]]),
                    point3=np.array([[1550], [4000]]),
                    point4=np.array([[900], [4000]]),
                ),
                # APF_function_for_DQN.generate_boundary(
                #     np.array([[900.0], [3900]]), np.array([[1550], [3900]]),
                #     np.array([[1550], [4000]]), np.array([[900], [4000]]), 11
                # ),

                APF_function_for_DQN.generate_boundary(
                    shape='rectangle',
                    step=11,
                    point1=np.array([[2050.0], [3900]]),
                    point2=np.array([[2700], [3900]]),
                    point3=np.array([[2700], [4000]]),
                    point4=np.array([[2050], [4000]]),
                ),
                # APF_function_for_DQN.generate_boundary(
                #     np.array([[2050.0], [3900]]), np.array([[2700], [3900]]),
                #     np.array([[2700], [4000]]), np.array([[2050], [4000]]), 11
                # )
            ]

        # fill obstacles info
        selected_obstacles = random.sample(obstacles, k=min(self.obstacle_num, len(obstacles)))
        obstacle_list = selected_obstacles
        self.obstacle_list = obstacle_list
        # print(len(obstacles), len(selected_obstacles))

        # compose for obstacle matrix
        if obstacle_list:
            self.obstacle_total = np.hstack([self.boundary] + obstacle_list)
        else:
            # no obstacles in environment
            self.obstacle_total = self.boundary

    def reset(self):
        # agent features
        self.agent_orientation_last = np.zeros((2, self.num_agent))
        self.agent_orientation_origin = np.zeros((2, self.num_agent))
        self.wall_following = np.zeros((1, self.num_agent))
        self.agent_orientation = np.zeros((2, self.num_agent))
        self.agent_position = np.zeros((2, self.num_agent))
        self.agent_position_last = np.zeros((2, self.num_agent))
        self.agent_position_origin = np.zeros((2, self.num_agent))
        self.obstacle_closest = []
        self.state = np.zeros((self.num_state, self.num_agent))
        self.global_state = np.zeros((self.num_state, self.num_agent))
        # evader features
        self.capture = np.zeros((1, self.num_evader))
        self.target_position = np.zeros((2, self.num_evader))
        self.target_position_last = np.zeros((2, self.num_evader))
        self.target_orientation = np.zeros((2, self.num_evader))
        self.escaper_slip_flag = [0] * self.num_evader
        self.escaper_wall_following = [0] * self.num_evader
        self.escaper_zigzag_flag = [0] * self.num_evader
        self.last_e = np.zeros((2, self.num_evader))
        self.zigzag_count = [0] * self.num_evader
        self.zigzag_last = np.zeros((2, self.num_evader))

        self.t = 0
        if self.task == 'aht':
            1/0
            done = np.ones((1, self.num_agent))
            count = 0
            while (done > 0).any():
                # print(done)
                # for idx in range(self.num_evader):
                #     self.target_position[0, idx] = random.randint(100, 3500)
                #     self.target_position[1, idx] = random.randint(4200, 5000)
                self.target_position[0, :] = np.random.random() * 2400 + 200
                self.target_position[1, :] = np.random.random() * 600 + 4200
                self.target_position = self.target_position + np.array(
                    [[i * 400 for i in range(self.num_evader)], [0] * self.num_evader])
                self.target_orientation = np.vstack((np.zeros((1, self.num_evader)), np.ones((1, self.num_evader))))
                # initialize pursuers' positions and headings

                for idx in range(self.num_agent):
                    self.agent_position[0, idx] = random.randint(20, 900) + 900 * idx
                    self.agent_position[1, idx] = random.randint(10, 1000)
                # self.agent_position = self.agent_position + np.array(
                #     [[i * 600 for i in range(self.num_agent)], [0] * self.num_agent])
                self.agent_orientation = np.vstack((np.zeros((1, self.num_agent)), np.ones((1, self.num_agent))))
                self.update_feature()
                self.update_state()  # update environment's state
                _, done = self.reward()
                count += 1
            # print(count)
        else:
            if self.num_agent == 3:
                # initialize evader's positions and headings
                self.target_position[0, :] = np.random.random() * 2400 + 200
                self.target_position[1, :] = np.random.random() * 600 + 4200
                self.target_position = self.target_position + np.array([[0, 400], [0, 0]])
                self.target_orientation = np.vstack((np.zeros((1, self.num_evader)), np.ones((1, self.num_evader))))
                # initialize pursuers' positions and headings
                self.agent_position[0, :] = np.random.random() * 2400 + 200
                self.agent_position[1, :] = np.random.random() * 600 + 200
                self.agent_position = self.agent_position + np.array(
                    [[i * 300 for i in range(self.num_agent)], [0] * self.num_agent])
                self.agent_orientation = np.vstack((np.zeros((1, self.num_agent)), np.ones((1, self.num_agent))))
            elif self.num_agent == 4:
                # initialize evader's positions and headings
                self.target_position[0, :] = np.random.random() * 2400 + 200
                self.target_position[1, :] = np.random.random() * 600 + 4200
                self.target_position = self.target_position + np.array(
                    [[i * 400 for i in range(self.num_evader)], [0] * self.num_evader])
                self.target_orientation = np.vstack((np.zeros((1, self.num_evader)), np.ones((1, self.num_evader))))
                # initialize pursuers' positions and headings
                self.agent_position[0, :] = np.random.random() * 1000 + 200
                self.agent_position[1, :] = np.random.random() * 600 + 200
                self.agent_position = self.agent_position + np.array(
                    [[i * 600 for i in range(self.num_agent)], [0] * self.num_agent])
                self.agent_orientation = np.vstack((np.zeros((1, self.num_agent)), np.ones((1, self.num_agent))))

        self.agent_position_origin = self.agent_position  # original positions
        self.agent_orientation_origin = self.agent_orientation  # original headings

        self.update_feature()
        self.update_state()  # update environment's state

        info = {"capture": self.capture,
                "d3qn_reward": 0,
                "d3qn_state": np.zeros((2 + 2 * self.num_evader + 1 + (self.num_agent - 1) * 2, self.num_agent))}
        return copy.deepcopy(self.state), copy.deepcopy(self.global_state), info

    def reward(self):
        reward = np.zeros((1, self.num_agent))  # reward buffer
        done = np.zeros((1, self.num_agent))  # done buffer
        capture_flag = False
        for j in range(self.num_evader):
            for i in range(self.num_agent):
                if np.linalg.norm(self.agent_position[:, i:i + 1] - self.target_position[:, j:j + 1]) < 300 and \
                        self.capture[0, j] == 0:
                    capture_flag = True  # r_main
                    self.capture[0, j] = 1
        if capture_flag:
            reward += 10
        if np.all(self.capture):
            done[:] = 1
        elif self.t == 1000:
            done[:] = 2.
        for i in range(self.num_agent):
            # pursuers failed though no collision
            if np.linalg.norm(
                    self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) < 100:
                # if the distance from the nearest obstacle is less than 100 mm
                reward[0, i] += self.neg_reward
                done[0, i] = 3
            if self.more_r:
                all_mate_distance = np.linalg.norm(self.agent_position[:, i:i + 1] - self.agent_position,
                                                   axis=0)
                for index, mate_distance in enumerate(all_mate_distance):
                    if mate_distance != 0:
                        if mate_distance > 200 and mate_distance < 251:
                            # very dangerous distance
                            # distance_from_mate_last = np.linalg.norm(
                            # self.agent_position_last[:, index:index + 1] - self.agent_position_last[:, i:i + 1])
                            # distance_from_mate_now = np.linalg.norm(
                            # self.agent_position[:, index:index + 1] - self.agent_position[:, i:i + 1])
                            # penalty_reward = (distance_from_mate_last - distance_from_mate_now) / 300
                            # # when penalty_reward > 0, the drone flys toward closest teamamtes then penalty it.
                            # if penalty_reward > 0:
                            reward[0, i] += self.neg_reward * 0.1
                        elif mate_distance > 250 and mate_distance < 301:
                            reward[0, i] += self.neg_reward * 0.01
                        elif mate_distance > 300 and mate_distance < 401:
                            reward[0, i] += self.neg_reward * 0.001

            distance_nearest_teamate = np.amin(
                np.linalg.norm(self.agent_position[:, i:i + 1] - np.delete(self.agent_position, i, axis=1),
                               axis=0))
            if distance_nearest_teamate < 200:
                # if the distance from the nearest teammate exceeds 200 mm
                reward[0, i] += self.neg_reward
                done[0, i] = 3
            for j in range(self.num_evader):
                if self.capture[0, j] == 0:
                    distance_from_target_last = np.linalg.norm(
                        self.target_position_last[:, j:j + 1] - self.agent_position_last[:, i:i + 1])
                    distance_from_target = np.linalg.norm(
                        self.target_position[:, j:j + 1] - self.agent_position[:, i:i + 1])
                    reward[0, i] += (distance_from_target_last - distance_from_target) / 300

        # in the training mode, initialize the collided pursuer, in validation mode, do nothing
        if np.any(done == 3):
            done[done == 0] = 2
        return reward, done

    def cal_d3qn_reward(self):
        reward = np.zeros((1, self.num_agent))  # reward buffer
        done = np.zeros((1, self.num_agent))  # done buffer
        capture_flag = False
        for j in range(self.num_evader):
            for i in range(self.num_agent):
                if np.linalg.norm(self.agent_position[:, i:i + 1] - self.target_position[:, j:j + 1]) < 300 and \
                        self.capture[0, j] == 0:
                    capture_flag = True  # r_main
                    self.capture[0, j] = 1
        if capture_flag:
            reward += 20
        if np.all(self.capture):
            done[:] = 1
        elif self.t == 1000:
            done[:] = 2.
        for i in range(self.num_agent):
            if np.arccos(np.clip(np.dot(np.ravel(self.agent_orientation_last[:, i:i + 1]),
                                        np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                self.agent_orientation_last[:, i:i + 1]) / np.linalg.norm(self.agent_orientation[:, i:i + 1]),
                                 -1, 1)) > np.radians(45):  # if the pursuer's steering angle exceeds 45
                reward[0, i] += -5
            # pursuers failed though no collision
            if np.linalg.norm(
                    self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) < 100:
                # if the distance from the nearest obstacle is less than 100 mm
                reward[0, i] += self.neg_reward
                done[0, i] = 3
            elif np.linalg.norm(
                    self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) > 150:
                # if the distance from the nearest obstacle is less than 100 mm
                reward[0, i] += 0
                done[0, i] = 3
            else:
                reward[0, i] += -2
                done[0, i] = 3

            if np.amin(np.linalg.norm(self.agent_position[:, i:i + 1] - np.delete(self.agent_position, i, axis=1),
                                      axis=0)) < 200:
                # if the distance from the nearest teammate exceeds 200 mm
                reward[0, i] += self.neg_reward
                done[0, i] = 3
            for j in range(self.num_evader):
                if self.capture[0, j] == 0:
                    distance_from_target_last = np.linalg.norm(
                        self.target_position_last[:, j:j + 1] - self.agent_position_last[:, i:i + 1])
                    distance_from_target = np.linalg.norm(
                        self.target_position[:, j:j + 1] - self.agent_position[:, i:i + 1])
                    reward[0, i] += (distance_from_target_last - distance_from_target) / 200

        # # in the training mode, initialize the collided pursuer, in validation mode, do nothing
        # if np.any(done == 3):
        #     done[done == 0] = 2
        return reward, done

    def step(self, all_actions, mate_types, attention_score_array=np.array([0]), ctrl_num=None):
        ctrl_num = self.ctrl_num
        if attention_score_array.size == 1:
            attention_score_array = np.zeros((self.num_agent, self.num_agent - 1))
        self.t += 1
        self.update_feature_last()
        ######agent#########
        F = np.zeros((2, self.num_agent))
        agent_position_buffer = np.zeros((2, self.num_agent))
        for i in range(self.num_agent):
            if i < ctrl_num:
                agent_type = self.algorithm
            else:
                agent_type = mate_types[i - ctrl_num].name
            F[:, i: i + 1] = self.transfer_action_to_final_F(all_actions, attention_score_array, i, agent_type)
            # if the pursuer is active, calculate its displacement
            # F[:, i:i + 1] * self.v vel
            agent_position_buffer[:, i:i + 1] = F[:, i:i + 1] * self.v * self.delta_t

        #######escaper########
        # calculate the evader's displacement according to the escaping policy
        F_escaper = np.zeros((2, self.num_evader))
        for i in range(self.num_evader):
            if self.capture[0, i] == 0:
                virtual_obstacle = np.zeros((2, 0))  # virtual obstacle buffer
                friends_position = np.delete(self.target_position, i, axis=1)
                for j in range(self.num_evader - 1):
                    temp = np.arange(1, 36)
                    temp = np.vstack((friends_position[0, j] + 100 * np.cos(np.radians(temp * 10)),
                                      friends_position[1, j] + 100 * np.sin(np.radians(temp * 10))))
                    virtual_obstacle = np.hstack((virtual_obstacle, temp))
                F_escaper[:, i:i + 1], self.zigzag_count[i], self.zigzag_last[:, i:i + 1], self.escaper_zigzag_flag[i], \
                    self.escaper_wall_following[i], self.escaper_slip_flag[i], distance_from_nearest_obstacle, \
                    self.last_e[:, i:i + 1] = escaper(self.agent_position,
                                                      self.target_position[:, i:i + 1],
                                                      self.target_orientation[:, i:i + 1],
                                                      np.hstack((self.obstacle_total, virtual_obstacle)),
                                                      self.num_agent, self.zigzag_count[i],
                                                      self.zigzag_last[:, i:i + 1], self.last_e[:, i:i + 1],
                                                      self.escaper_slip_flag[i],
                                                      v_max=self.e_v)

        #####update#####
        # rewrite the pursuer position by the pos from real world
        self.agent_position = self.agent_position + agent_position_buffer  # update pursuers'positions

        self.agent_orientation = F  # update pursuers' headings

        # rewrite the pursuer position by the pos from real world
        self.target_position = self.target_position + F_escaper * self.delta_t  # update the evader's position

        self.target_orientation = F_escaper  # update the evader's heading

        reward, done = self.reward()  # calculate reward function
        self.done = done
        self.update_feature()
        self.update_state()  # update environment's state

        info = {"d3qn_state": self.cal_d3qn_state(), "d3qn_reward_and_done": self.cal_d3qn_reward(),
                "capture": self.capture}
        return self.state, self.global_state, reward, done, info

    def update_state(self):
        self.state = np.zeros((self.num_state, self.num_agent))  # clear the environment state
        self.global_state = np.zeros((self.num_state, self.num_agent))  # 全局状态

        # clear the nearest target
        for i in range(self.num_agent):
            # the distance form the nearest target
            for j in range(self.num_evader):
                temp = self.target_position[:, j:j + 1] - self.agent_position[:, i:i + 1]
                # the bearing of evader
                angle = np.arccos(
                    np.clip(
                        np.dot(np.ravel(temp), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                            temp) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1))
                if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp)) > 0:
                    pass
                else:
                    angle = -angle
                if np.linalg.norm(temp) < self.r_perception or self.old_env:
                    # if within the observable area
                    self.state[j * 2, i] = np.linalg.norm(temp) / 5000
                    self.state[j * 2 + 1, i] = angle / np.pi
                else:
                    self.state[j * 2, i] = 2
                    self.state[j * 2 + 1, i] = 0

                # 更新全局状态
                self.global_state[j * 2, i] = np.linalg.norm(temp) / 5000
                self.global_state[j * 2 + 1, i] = angle / np.pi

                # 如果目标被捕获
                if self.capture[0, j] == 1:
                    self.state[j * 2, i] = 2
                    self.state[j * 2 + 1, i] = 0
                    self.global_state[j * 2, i] = 2
                    self.global_state[j * 2 + 1, i] = 0

            temp = self.obstacle_closest[:, i:i + 1] - self.agent_position[:, i:i + 1]
            # the bearing of the nearest obstacle
            angle = np.arccos(
                np.clip(
                    np.dot(np.ravel(temp), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                        temp) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1))
            if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp)) > 0:
                pass
            else:
                angle = -angle

            if np.linalg.norm(temp) < self.r_perception or self.old_env:
                # if within the observable area
                self.state[self.num_evader * 2, i] = np.linalg.norm(temp) / 5000
                self.state[self.num_evader * 2 + 1, i] = angle / np.pi
            else:
                self.state[self.num_evader * 2, i] = 2
                self.state[self.num_evader * 2 + 1, i] = 0

            self.global_state[self.num_evader * 2, i] = self.state[self.num_evader * 2, i]
            self.global_state[self.num_evader * 2 + 1, i] = self.state[self.num_evader * 2 + 1, i]

            # the distance from evader
            friends_position = np.delete(self.agent_position, i, axis=1)  # teammate positions
            for j in range(self.num_agent - 1):
                temp = friends_position[:, j:j + 1] - self.agent_position[:, i:i + 1]
                angle = np.arccos(
                    np.clip(
                        np.dot(np.ravel(temp), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                            temp) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1))
                if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp)) > 0:
                    pass
                else:
                    angle = -angle
                # mask distant teammates and update state
                # print(friends_position[:, j:j + 1],self.agent_position[:, i:i + 1], np.linalg.norm(temp), self.r_perception)
                if np.linalg.norm(temp) < self.r_perception:
                    self.state[2 * self.num_evader + 2 + 2 * j, i] = np.linalg.norm(temp) / 5000
                    self.state[2 * self.num_evader + 2 + 2 * j + 1, i] = angle / np.pi
                else:
                    self.state[2 * self.num_evader + 2 + 2 * j, i] = 2
                    self.state[2 * self.num_evader + 2 + 2 * j + 1, i] = 0

                self.global_state[2 * self.num_evader + 2 + 2 * j, i] = np.linalg.norm(temp) / 5000
                self.global_state[2 * self.num_evader + 2 + 2 * j + 1, i] = angle / np.pi

    def update_feature(self):
        virtual_obstacle = np.zeros((2, 0))  # virtual obstacle buffer
        for i in range(self.num_evader):
            if self.capture[0, i] == 1:  # if any pursuer is inactive
                temp = np.arange(1, 36)
                temp = np.vstack((self.target_position[0, i] + 100 * np.cos(np.radians(temp * 10)),
                                  self.target_position[1, i] + 100 * np.sin(np.radians(temp * 10))))
                virtual_obstacle = np.hstack((virtual_obstacle, temp))
        # add virtual obstacles into obstacles list
        obstacle_with_target = np.hstack((self.obstacle_total, virtual_obstacle))
        if self.num_agent == 3:
            obstacle_closest_with_target = np.zeros((2, self.num_agent))
            for i in range(self.num_agent):
                # the index of nearest obstacle (considering virtual obstacles)
                temp = np.argmin(np.linalg.norm(obstacle_with_target - self.agent_position[:, i:i + 1], axis=0))
                # the position of nearest obstacle (considering virtual obstacles)
                obstacle_closest_with_target[:, i:i + 1] = obstacle_with_target[:, temp:temp + 1]
            self.obstacle_closest = obstacle_closest_with_target
        elif self.num_agent > 3:
            obstacle_closest_with_target_and_friends = np.zeros((2, self.num_agent))
            for i in range(self.num_agent):
                friends_position = np.delete(self.agent_position, i, axis=1)  # teammate positions
                obstacle_with_target_and_friends = np.hstack((obstacle_with_target, friends_position))
                # the index of nearest obstacle (considering virtual obstacles)
                temp = np.argmin(
                    np.linalg.norm(obstacle_with_target_and_friends - self.agent_position[:, i:i + 1], axis=0))
                # the position of nearest obstacle (considering virtual obstacles)
                obstacle_closest_with_target_and_friends[:, i:i + 1] = obstacle_with_target_and_friends[:,
                                                                       temp:temp + 1]
            self.obstacle_closest = obstacle_closest_with_target_and_friends

    def update_feature_last(self):
        self.agent_position_last = copy.deepcopy(self.agent_position)
        self.agent_orientation_last = copy.deepcopy(self.agent_orientation)
        self.target_position_last = copy.deepcopy(self.target_position)

    def from_action_to_APF(self, action, attention_score_array, which=0):
        if np.all(which == 0):
            which = np.array(range(self.num_agent)).reshape(1, -1)
        scale_repulse = np.zeros_like(action)  # eta buffer
        individual_balance = np.zeros_like(action)  # lambda buffer
        for j in range(action.shape[0]):
            for i in range(action.shape[1]):  # transform the action indexes into parameter pairs
                if action[j, i] < 8:
                    scale_repulse[j, i] = 1e6
                elif action[j, i] < 16:
                    scale_repulse[j, i] = 1e7
                else:
                    scale_repulse[j, i] = 1e8
                if action[j, i] % 8 == 0:
                    individual_balance[j, i] = 4000 / 7 * 0
                elif action[j, i] % 8 == 1:
                    individual_balance[j, i] = 4000 / 7 * 1
                elif action[j, i] % 8 == 2:
                    individual_balance[j, i] = 4000 / 7 * 2
                elif action[j, i] % 8 == 3:
                    individual_balance[j, i] = 4000 / 7 * 3
                elif action[j, i] % 8 == 4:
                    individual_balance[j, i] = 4000 / 7 * 4
                elif action[j, i] % 8 == 5:
                    individual_balance[j, i] = 4000 / 7 * 5
                elif action[j, i] % 8 == 6:
                    individual_balance[j, i] = 4000 / 7 * 6
                elif action[j, i] % 8 == 7:
                    individual_balance[j, i] = 4000 / 7 * 7
        F, wall_following = APF_function_for_DQN.total_decision(self.agent_position,
                                                                self.agent_orientation,
                                                                self.obstacle_closest,
                                                                self.target_position,
                                                                scale_repulse,
                                                                individual_balance,
                                                                self.r_perception, which, attention_score_array)
        return F, wall_following

    def transfer_action_to_final_F(self, all_actions, attention_score_array, index, current_agent_type):
        if current_agent_type in ["PPO", "IPPO", "MAPPO", 'POAM']:
            c_action = np.clip(all_actions, -1, 1)
            agent_orientation_angle = np.arctan2(self.agent_orientation[1, :], self.agent_orientation[0, :]).reshape(1,
                                                                                                                     self.num_agent)
            temp1 = np.cos(np.radians(c_action * 45) + agent_orientation_angle)
            temp2 = np.sin(np.radians(c_action * 45) + agent_orientation_angle)
            force = np.vstack((temp1, temp2))
            return force[:, index:index + 1]
        elif current_agent_type in ["DQN", "D3QN"]:
            force, wall_following = self.from_action_to_APF(all_actions, attention_score_array)
            force = np.round(force * 1000) / 1000
            force = np.squeeze(force, axis=2)

            f_apf = copy.deepcopy(force[:, index:index + 1])
            agent_orientation = self.agent_orientation[:, index:index + 1]
            temp = np.radians(45)
            if np.arccos(np.clip(np.dot(np.ravel(agent_orientation), np.ravel(f_apf)) / np.linalg.norm(
                    agent_orientation) / np.linalg.norm(f_apf), -1, 1)) > temp:
                rotate_matrix = np.array([[np.cos(temp), -np.sin(temp)], [np.sin(temp), np.cos(temp)]])
                temp1 = np.matmul(rotate_matrix, agent_orientation)
                rotate_matrix = np.array(
                    [[np.cos(-temp), -np.sin(-temp)], [np.sin(-temp), np.cos(-temp)]])
                temp2 = np.matmul(rotate_matrix, agent_orientation)
                if np.dot(np.ravel(temp1), np.ravel(f_apf)) > np.dot(np.ravel(temp2), np.ravel(f_apf)):
                    f_apf = temp1
                else:
                    f_apf = temp2
            return copy.deepcopy(f_apf)
        elif current_agent_type in ["DACOOP"]:
            force, wall_following = self.from_action_to_APF(all_actions, attention_score_array)
            force = np.round(force * 1000) / 1000
            force = np.squeeze(force, axis=2)

            f_apf = copy.deepcopy(force[:, index:index + 1])
            agent_orientation = self.agent_orientation[:, index:index + 1]
            temp = np.radians(30)
            if np.arccos(np.clip(np.dot(np.ravel(agent_orientation), np.ravel(f_apf)) / np.linalg.norm(
                    agent_orientation) / np.linalg.norm(f_apf), -1, 1)) > temp:
                rotate_matrix = np.array([[np.cos(temp), -np.sin(temp)], [np.sin(temp), np.cos(temp)]])
                temp1 = np.matmul(rotate_matrix, agent_orientation)
                rotate_matrix = np.array(
                    [[np.cos(-temp), -np.sin(-temp)], [np.sin(-temp), np.cos(-temp)]])
                temp2 = np.matmul(rotate_matrix, agent_orientation)
                if np.dot(np.ravel(temp1), np.ravel(f_apf)) > np.dot(np.ravel(temp2), np.ravel(f_apf)):
                    f_apf = temp1
                else:
                    f_apf = temp2
            if np.isnan(f_apf).any():
                print('nan in F')
                f_apf = np.zeros((2, 1))
            self.wall_following = np.squeeze(wall_following, axis=2)
            return copy.deepcopy(f_apf)

        elif current_agent_type in ["VICSEK"]:
            c_target_position = copy.deepcopy(self.target_position)
            c_target_orientation = copy.deepcopy(self.target_orientation)
            if np.any(self.capture == 1):
                for i in range(self.num_evader - 1, -1, -1):
                    if self.capture[0, i] == 1:
                        c_target_position = np.delete(c_target_position, i, axis=1)
                        c_target_orientation = np.delete(c_target_orientation, i, axis=1)

            c_target_position = c_target_position[:, :1]
            c_target_orientation = c_target_orientation[:, :1]
            return AgentVicsek.vicsek_chaser_velocity(self.agent_position, self.agent_orientation * self.v, index,
                                                      c_target_position, c_target_orientation,
                                                      self.obstacle_closest[:, index:index + 1])
        elif current_agent_type in ["GREEDY"]:
            return AgentGreedy.greedy_chaser_velocity(self.state[:, index:index + 1],
                                                      self.agent_orientation[:, index:index + 1],
                                                      self.num_agent, self.num_evader, self.capture)

        else:
            raise ValueError('algorithm not supported in environment.py')

    def cal_d3qn_state(self):
        d3qn_state = np.zeros(
            (2 + 2 * self.num_evader + 1 + (self.num_agent - 1) * 2, self.num_agent))  # clear the environment state
        # clear the nearest obstacle list(considering virtual obstacles)
        for i in range(self.num_agent):
            # the distance form the nearest obstacle
            temp1 = self.obstacle_closest[:, i:i + 1] - self.agent_position[:, i:i + 1]
            # the bearing of the nearest obstacle
            angle1 = np.arccos(
                np.clip(
                    np.dot(np.ravel(temp1), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                        temp1) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1)) / np.pi
            if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp1)) > 0:
                pass
            else:
                angle1 = -angle1
            # the distance from evader
            temp_obstacle_evader = [np.linalg.norm(temp1) / 5000, angle1]
            for e_ind in range(self.num_evader):
                temp2 = self.target_position[0][e_ind] - self.agent_position[:, i:i + 1]
                # the bearing of evader
                angle2 = np.arccos(
                    np.clip(
                        np.dot(np.ravel(temp2), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                            temp2) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1)) / np.pi
                if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp2)) > 0:
                    pass
                else:
                    angle2 = -angle2
                temp_obstacle_evader.extend([np.linalg.norm(temp2) / 5000, angle2])

            state = np.zeros((2 + 2 * self.num_evader + 1 + (self.num_agent - 1) * 2,))  # state buffer
            state[:2 * self.num_evader + 2] = np.array(temp_obstacle_evader,
                                                       dtype='float32')  # update state

            friends_position = np.delete(self.agent_position, i, axis=1)  # teammate positions
            for j in range(self.num_agent - 1):
                friend_position = friends_position[:, j:j + 1]  # teammate postion
                self_position = self.agent_position[:, i:i + 1]
                self_orientation = self.agent_orientation[:, i:i + 1]
                temp = friend_position - self_position
                distance = np.linalg.norm(temp)  # the distance from teammate
                # the bearing of teammate
                angle = np.arccos(
                    np.clip(
                        np.dot(np.ravel(temp), np.ravel(self_orientation)) / distance / np.linalg.norm(
                            self_orientation), -1, 1)) / np.pi
                if np.cross(np.ravel(self_orientation), np.ravel(temp)) > 0:
                    pass
                else:
                    angle = -angle
                # mask distant teammates and update state
                if distance < self.r_perception:
                    state[2 * self.num_evader + 2 + 1 + 2 * j] = np.linalg.norm(temp) / 5000
                    state[2 * self.num_evader + 2 + 2 + 2 * j] = np.array(angle)
                else:
                    state[2 * self.num_evader + 2 + 1 + 2 * j] = 2
                    state[2 * self.num_evader + 2 + 2 + 2 * j] = 0
            if np.any(self.done == 1):
                state[2 * self.num_evader + 2] = 1
            else:
                state[2 * self.num_evader + 2] = 0
            d3qn_state[:, i] = state
        return d3qn_state

    def render(self, buffer=None):
        plt.figure(1, figsize=(4, 5))
        # The code is clearing the current axes in a matplotlib figure.
        plt.cla()
        ax = plt.gca()
        # plt.xlim([-100, 3700])
        # plt.ylim([-100, 5100])
        ax.set_aspect(1)
        ax.set_axis_off()
        # plot obstacles and boundary
        plt.plot(self.boundary[0, :], self.boundary[1, :], 'black')
        for i, obstacle in enumerate(self.obstacle_list, start=1):
            plt.fill(obstacle[0, :], obstacle[1, :], color='gray')
        # plot evader
        for i in range(self.num_evader):
            # circle = mpatches.Circle(np.ravel(self.target_position[:, i:i + 1]), 100, facecolor=color)
            # ax.add_patch(circle)
            drone_img = mpimg.imread('./env/assets/drone_black.png')  # Make sure to provide the correct path
            drone_imagebox = OffsetImage(drone_img, zoom=0.8)  # Adjust zoom as necessary
            drone_ab = AnnotationBbox(drone_imagebox, np.ravel(self.target_position[:, i:i + 1]), frameon=False)

            ax.add_artist(drone_ab)
        # plot pursuers
        color = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(self.num_agent):
            # circle = mpatches.Circle(self.agent_position[:, i], 100, facecolor=color[i])
            # ax.add_patch(circle)

            drone_img = mpimg.imread('./env/assets/drone_red.png')  # Make sure to provide the correct path
            drone_imagebox = OffsetImage(drone_img, zoom=0.8)  # Adjust zoom as necessary
            drone_ab = AnnotationBbox(drone_imagebox, self.agent_position[:, i], frameon=False)

            ax.add_artist(drone_ab)
            # if self.wall_following[0, i]:
            #     plt.quiver(self.agent_position[0, i], self.agent_position[1, i], self.agent_orientation[0, i],
            #                self.agent_orientation[1, i], color='green', scale=10)
            # else:
            length = (self.agent_orientation[0, i] ** 2 + self.agent_orientation[1, i] ** 2) ** 0.5 * 3
            plt.quiver(self.agent_position[0, i], self.agent_position[1, i], self.agent_orientation[0, i] / length,
                       self.agent_orientation[1, i] / length, color='blue', scale=5, width=0.005)
            
            plt.text(
                self.agent_position[0, i], 
                self.agent_position[1, i] - 170,
                f"{i}",
                fontsize=8,
                color="black",
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'),
                horizontalalignment="center",
                verticalalignment="center"
            )
        plt.tight_layout(pad=0)
        plt.savefig(f'./env/{self.num_agent}-vs-{self.num_evader}-{self.env_idx}.pdf', format='pdf', dpi=1200)
        if buffer is None:
            plt.show(block=False)
            plt.pause(0.001)
            return
        else:
            buf = io.BytesIO()  # Create a buffer to save the figure
            plt.savefig(buf, format='png')  # Save figure to buffer
            buf.seek(0)
            buffer.append(buf)
            return buffer


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    config = {
        "num_action": 24,
        "num_state": 4 + 1 + (3 - 1) * 2,
        "num_p": 3,
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
    env = environment(config)
    observations, info = env.reset()
    env.render()  # render the pursuit process
    # The code is saving the current figure in matplotlib as an image file named "test_env.png".
    plt.savefig("test_env.png")
