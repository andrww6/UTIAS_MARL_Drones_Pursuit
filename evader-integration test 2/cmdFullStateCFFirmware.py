#!/usr/bin/env python
"""
Usage: 
    python cmdFullStateCFFirmware.py <path/to/controller.py> config.yaml

"""
import pickle as pkl
import os, sys
cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, '../'))

import argparse, json, yaml
import time
from math import atan2, asin

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pycrazyswarm import *
import rospy
from geometry_msgs.msg import TransformStamped
# from geometry_msgs.msg import PoseStamped

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

class ViconWatcher:
    def __init__(self):
        # rospy.init_node("playback_node")
        self.vicon_sub = rospy.Subscriber("/vicon/cf9/cf9", TransformStamped, self.vicon_callback)
        # self.vicon_sub = rospy.Subscriber("/cf10", TransformStamped, self.vicon_callback)
        # self.vicon_sub = rospy.Subscriber("/vrpn_client_node/cf9/pose", PoseStamped, self.vicon_callback)
        self.pos = None
        self.rpy = None

    def vicon_callback(self, data):
        self.child_frame_id = data.child_frame_id
        self.pos = np.array([
            data.transform.translation.x,
            data.transform.translation.y,
            data.transform.translation.z,
        ])
        rpy = euler_from_quaternion(
            data.transform.rotation.x,
            data.transform.rotation.y,
            data.transform.rotation.z,
            data.transform.rotation.w,
        )
        self.rpy = np.array(rpy)

# dsl25@dsl25-ThinkPad-P52:~$ rostopic echo /vi
# /vicon/cf9/cf9               /vicon/cf_lo2/cf_lo2         /vicon/cf_obs4/cf_obs4
# /vicon/cf_hi1/cf_hi1         /vicon/cf_obs1/cf_obs1       /vicon/markers
# /vicon/cf_hi2/cf_hi2         /vicon/cf_obs2/cf_obs2       /virtual_interactive_object
# /vicon/cf_lo1/cf_lo1         /vicon/cf_obs3/cf_obs3

class ObjectWatcher:
    def __init__(self, object: str=""):
        self.vicon_sub = rospy.Subscriber("/vicon/"+object+"/"+object, TransformStamped, self.vicon_callback)
        # self.vicon_sub = rospy.Subscriber("/vrpn_client_node/cf9/pose", PoseStamped, self.vicon_callback)
        self.pos = None
        self.rpy = None

    def vicon_callback(self, data):
        self.child_frame_id = data.header.frame_id
        self.pos = np.array([
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z,
        ])
        rpy = euler_from_quaternion(
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w,
        )
        self.rpy = np.array(rpy)

def load_controller(path):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    try:
        return controller_module.Controller, controller_module.Command
    except ImportError as e:
        raise e

def eval_token(token):
    """Converts string token to int, float or str.

    """
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except TypeError:
        return token

# def read_file(file_path, sep=","):
#     """Loads content from a file (json, yaml, csv, txt).
    
#     For json & yaml files returns a dict.
#     Ror csv & txt returns list of lines.

#     """
#     if len(file_path) < 1 or not os.path.exists(file_path):
#         return None
#     # load file
#     f = open(file_path, "r")
#     if "json" in file_path:
#         data = json.load(f)
#     elif "yaml" in file_path:
#         data = yaml.load(f, Loader=yaml.FullLoader)
#     else:
#         sep = sep if "csv" in file_path else " "
#         data = []
#         for line in f.readlines():
#             line_post = [eval_token(t) for t in line.strip().split(sep)]
#             # if only sinlge item in line
#             if len(line_post) == 1:
#                 line_post = line_post[0]
#             if len(line_post) > 0:
#                 data.append(line_post)
#     f.close()
#     return data


if __name__ == "__main__":
    SCRIPT_START = time.time() 
    parser = argparse.ArgumentParser()
    parser.add_argument("controller", type=str, help="path to controller file")
    parser.add_argument("config", type=str, help="path to course configuration file")
    parser.add_argument("--overrides", type=str, help="path to environment configuration file")
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.controller))

    Controller, Command = load_controller(args.controller)
    
    swarm = Crazyswarm("../../launch/crazyflies.yaml")
    # swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    vicon = ViconWatcher()

    hi1 = ObjectWatcher("cf_hi1")
    hi2 = ObjectWatcher("cf_hi2")
    lo1 = ObjectWatcher("cf_lo1")
    lo2 = ObjectWatcher("cf_lo2")
    obs1 = ObjectWatcher("cf_obs1")
    obs2 = ObjectWatcher("cf_obs2")
    obs3 = ObjectWatcher("cf_obs3")
    obs4 = ObjectWatcher("cf_obs4")

    timeout = 10
    
    while vicon.pos is None or vicon.rpy is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')

    init_pos = vicon.pos
    init_rpy = vicon.rpy
    print("Vicon found.")
    

    """
    timeout = 10
    while hi1.pos is None or hi1.rpy is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')
    hi1_pos = hi1.pos
    hi1_rpy = hi1.rpy

    timeout = 10
    while hi2.pos is None or hi2.rpy is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')
    hi2_pos = hi2.pos
    hi2_rpy = hi2.rpy

    timeout = 10
    while lo1.pos is None or lo1.rpy is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')
    lo1_pos = lo1.pos
    lo1_rpy = lo1.rpy

    timeout = 10
    while lo2.pos is None or lo2.rpy is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')
    lo2_pos = lo2.pos
    lo2_rpy = lo2.rpy

    timeout = 10
    while obs1.pos is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')
    obs1_pos = obs1.pos

    timeout = 10
    while obs2.pos is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')
    obs2_pos = obs2.pos

    timeout = 10
    while obs3.pos is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')
    obs3_pos = obs3.pos

    timeout = 10
    while obs4.pos is None:
        print("Waiting for vicon...")
        timeout -= 1
        time.sleep(1)
        if not timeout:
            raise TimeoutError('Vicon unavailable.')
    obs4_pos = obs4.pos

    nominal_gates_pos_and_type = [
        [hi1_pos[0], hi1_pos[1], 0] + [0,0,hi1_rpy[2]] + [0],
        [lo1_pos[0], lo1_pos[1], 0] + [0,0,lo1_rpy[2]] + [1],
        [lo2_pos[0], lo2_pos[1], 0] + [0,0,lo2_rpy[2]] + [1],
        [hi2_pos[0], hi2_pos[1], 0] + [0,0,hi2_rpy[2]] + [0]
    ]

    nominal_obstacles_pos = [
        [obs1_pos[0], obs1_pos[1], 0]  + [0, 0, 0],
        [obs2_pos[0], obs2_pos[1], 0]  + [0, 0, 0],
        [obs3_pos[0], obs3_pos[1], 0]  + [0, 0, 0],
        [obs4_pos[0], obs4_pos[1], 0]  + [0, 0, 0]
    ]
    """
    
    nominal_gates_pos_and_type = [[0.5, -2.5, 0, 0, 0, -1.57, 0], [2, -1.5, 0, 0, 0, 0, 1], [0, 0.5, 0, 0, 0, 1.57, 1], [-0.5, 1.5, 0, 0, 0, 0, 0]]
    nominal_obstacles_pos = [[1.5, -2.5, 0, 0, 0, 0], [0.5, -1, 0, 0, 0, 0], [1.5, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]]
    print(nominal_gates_pos_and_type)
    print(nominal_obstacles_pos)

    # Create a safe-control-gym environment from which to take the symbolic models
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    config.quadrotor_config['ctrl_freq'] = 500
    env = make('quadrotor', **config.quadrotor_config)
    env_obs, env_info = env.reset()
    print(env_obs, env_info)

    # Override environment state and evaluate constraints
    env.state = [vicon.pos[0], 0, vicon.pos[1], 0, vicon.pos[2], 0, vicon.rpy[0], vicon.rpy[1], vicon.rpy[2], 0, 0, 0]
    cnstr_eval = env.constraints.get_values(env, only_state=True)

    init_info = {
        #
        'symbolic_model': env_info['symbolic_model'], # <safe_control_gym.math_and_models.symbolic_systems.SymbolicModel object at 0x7fac3a161430>,
        'nominal_physical_parameters': {'quadrotor_mass': 0.03454, 
                                        'quadrotor_ixx_inertia': 1.4e-05, 'quadrotor_iyy_inertia': 1.4e-05, 'quadrotor_izz_inertia': 2.17e-05},
        #
        'x_reference': [-0.5 ,  0.  ,  2.9 ,  0.  ,  0.75,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        'u_reference': [0.084623, 0.084623, 0.084623, 0.084623],
        #
        'symbolic_constraints': env_info['symbolic_constraints'], # [<function LinearConstraint.__init__.<locals>.<lambda> at 0x7fac49139160>, <function LinearConstraint.__init__.<locals>.<lambda> at 0x7fac3a14a5e0>],
        #
        'ctrl_timestep': 0.03333333333333333,
        'ctrl_freq': 30,
        'episode_len_sec': 33,
        'quadrotor_kf': 3.16e-10,
        'quadrotor_km': 7.94e-12,
        'gate_dimensions': {'tall': {'shape': 'square', 'height': 1.0, 'edge': 0.45}, 'low': {'shape': 'square', 'height': 0.525, 'edge': 0.45}},
        'obstacle_dimensions': {'shape': 'cylinder', 'height': 1.05, 'radius': 0.05},
        'nominal_gates_pos_and_type': nominal_gates_pos_and_type,
        'nominal_obstacles_pos': nominal_obstacles_pos,
        # 'nominal_gates_pos_and_type': [[0.5, -2.5, 0, 0, 0, -1.57, 0], [2, -1.5, 0, 0, 0, 0, 1], [0, 0.2, 0, 0, 0, 1.57, 1], [-0.5, 1.5, 0, 0, 0, 0, 0]],
        # 'nominal_obstacles_pos': [[1.5, -2.5, 0, 0, 0, 0], [0.5, -1, 0, 0, 0, 0], [1.5, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]],
        #
        'initial_state_randomization': env_info['initial_state_randomization'], # Munch({'init_x': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'init_y': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'init_z': Munch({'distrib': 'uniform', 'low': 0.0, 'high': 0.02}), 'init_phi': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'init_theta': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'init_psi': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1})}),
        'inertial_prop_randomization': env_info['inertial_prop_randomization'], # Munch({'M': Munch({'distrib': 'uniform', 'low': -0.01, 'high': 0.01}), 'Ixx': Munch({'distrib': 'uniform', 'low': -1e-06, 'high': 1e-06}), 'Iyy': Munch({'distrib': 'uniform', 'low': -1e-06, 'high': 1e-06}), 'Izz': Munch({'distrib': 'uniform', 'low': -1e-06, 'high': 1e-06})}),
        'gates_and_obs_randomization': env_info['gates_and_obs_randomization'], # Munch({'gates': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1}), 'obstacles': Munch({'distrib': 'uniform', 'low': -0.1, 'high': 0.1})}),
        'disturbances': env_info['disturbances'], # Munch({'action': [Munch({'disturbance_func': 'white_noise', 'std': 0.001})], 'dynamics': [Munch({'disturbance_func': 'uniform', 'low': [-0.1, -0.1, -0.1], 'high': [0.1, 0.1, 0.1]})]}),
        # COULD/SHOULD THESE BE None or {} INSTEAD?
        #
        'urdf_dir': None, # '/Users/jacopo/GitHub/beta-iros-competition/safe_control_gym/envs/gym_pybullet_drones/assets',
        'pyb_client': None, # 0,
        'constraint_values': cnstr_eval # [-2.03390077, -0.09345386, -0.14960551, -3.96609923, -5.90654614, -1.95039449]
    }

    CTRL_FREQ = init_info['ctrl_freq']

    # Create controller.
    vicon_obs = [init_pos[0], 0, init_pos[1], 0, init_pos[2], 0, init_rpy[0], init_rpy[1], init_rpy[2], 0, 0, 0]
    ctrl = Controller(vicon_obs, init_info, True)

    # Initial gate.
    current_target_gate_id = 0

    # ---- log data start
    # print("press button to takeoff ----- ")
    # swarm.input.waitUntilButtonPressed()    

    # ---- commands for log
    log_cmd = []; 

    completed = False
    print(f"Setup time: {time.time() - SCRIPT_START:.3}s")
    START_TIME = time.time() 
    while not timeHelper.isShutdown():
        time.sleep(0.1)  # TODO: CHECK THE FREQUENCY or limit it so commands dont flood the drone
        curr_time = time.time() - START_TIME

        done = False # Leave always false in sim2real

        # Override environment state and evaluate constraints
        env.state = [vicon.pos[0], 0, vicon.pos[1], 0, vicon.pos[2], 0, vicon.rpy[0], vicon.rpy[1], vicon.rpy[2], 0, 0, 0]
        state_error = (env.state - env.X_GOAL) * env.info_mse_metric_state_weight
        cnstr_eval = env.constraints.get_values(env, only_state=True)
        if env.constraints.is_violated(env, c_value=cnstr_eval):
            # IROS 2022 - Constrain violation flag for reward.
            env.cnstr_violation = True
            cnstr_num = 1
        else:
            # IROS 2022 - Constrain violation flag for reward.
            env.cnstr_violation = False
            cnstr_num = 0

        # This only looks at the x-y plane, could be improved
        gate_dist = np.sqrt(np.sum((vicon.pos[0:2] - nominal_gates_pos_and_type[current_target_gate_id][0:2])**2))
        #
        current_target_gate_in_range = True if gate_dist < 0.45 else False
        # current_target_gate_in_range = False # Sim2real difference, potentially affects solutions 

        info = { # TO DO
            'mse': np.sum(state_error ** 2),
            #
            'collision': (None, False), # Leave always false in sim2real
            #
            'current_target_gate_id': current_target_gate_id,
            'current_target_gate_in_range': current_target_gate_in_range,
            'current_target_gate_pos': nominal_gates_pos_and_type[current_target_gate_id][0:6], # "Exact" regardless of distance
            'current_target_gate_type': nominal_gates_pos_and_type[current_target_gate_id][6],
            #
            'at_goal_position': False, # Leave always false in sim2real
            'task_completed': False, # Leave always false in sim2real
            #
            'constraint_values': cnstr_eval, # array([-0.02496828, -0.08704742, -0.10894883, -0.04954095, -0.09521148, -0.03313234, -0.01123093, -0.07063881, -2.03338112, -0.09301162, -0.14799449, -3.96661888, -5.90698838, -1.95200551]),
            'constraint_violation': cnstr_num # 0
        }

        #####################################################################
        #####################################################################
        # This only looks at x-y plan and could be improved
        # if nominal_gates_pos_and_type[current_target_gate_id][6] == 0: # high gate
        #     if gate_dist < 0.175 and vicon.pos[2] > 0.85 and vicon.pos[2] < 1.25:
        #         current_target_gate_id += 1
        # elif nominal_gates_pos_and_type[current_target_gate_id][6] == 1: # low gate
        #     if gate_dist < 0.175 and vicon.pos[2] > 0.4 and vicon.pos[2] < 0.8:
        #         current_target_gate_id += 1
        # current_target_gate_id = min(current_target_gate_id, 3)
        # 
        # This is terribly ad-hoc for the competition scenario
        current_target_gate_pos = nominal_gates_pos_and_type[current_target_gate_id][0:3]
        if current_target_gate_id == 0:
            if vicon.pos[2] > 0.85 and vicon.pos[2] < 1.25 \
                    and vicon.pos[0] > current_target_gate_pos[0]-0.0 and vicon.pos[0] < current_target_gate_pos[0]+0.05 \
                    and vicon.pos[1] > current_target_gate_pos[1]-0.2 and vicon.pos[1] < current_target_gate_pos[1]+0.2:
                current_target_gate_id += 1
        elif current_target_gate_id == 1:
            if vicon.pos[2] > 0.4 and vicon.pos[2] < 0.8 \
                    and vicon.pos[1] > current_target_gate_pos[1]-0.0 and vicon.pos[1] < current_target_gate_pos[1]+0.05 \
                    and vicon.pos[0] > current_target_gate_pos[0]-0.2 and vicon.pos[0] < current_target_gate_pos[0]+0.2:
                current_target_gate_id += 1
        elif current_target_gate_id == 2:
            if vicon.pos[2] > 0.4 and vicon.pos[2] < 0.8 \
                    and vicon.pos[0] > current_target_gate_pos[0]-0.05 and vicon.pos[0] < current_target_gate_pos[0]+0.0 \
                    and vicon.pos[1] > current_target_gate_pos[1]-0.2 and vicon.pos[1] < current_target_gate_pos[1]+0.2:
                current_target_gate_id += 1
        elif current_target_gate_id == 3:
            if vicon.pos[2] > 0.85 and vicon.pos[2] < 1.25 \
                    and vicon.pos[1] > current_target_gate_pos[1]-0.0 and vicon.pos[1] < current_target_gate_pos[1]+0.05 \
                    and vicon.pos[0] > current_target_gate_pos[0]-0.2 and vicon.pos[0] < current_target_gate_pos[0]+0.2:
                current_target_gate_id = -1
                at_goal_time = time.time() 

        if current_target_gate_id == -1:
            goal_pos = np.array([env.X_GOAL[0], env.X_GOAL[2], env.X_GOAL[4]])
            print(f"{time.time() - at_goal_time:.4}s and {np.linalg.norm(vicon.pos[0:3] - goal_pos)}m away")
            if np.linalg.norm(vicon.pos[0:3] - goal_pos) >= 0.15:
                print(f"First hit goal position in {curr_time:.4}s")
                at_goal_time = time.time()
            elif time.time() - at_goal_time > 2:
                print(f"Task Completed in {curr_time:.4}s")
                completed = True
        #####################################################################
        #####################################################################

        #####################################################################
        #####################################################################
        reward = 0 # TO DO (or not needed for sim2real?)
        # # Reward for stepping through the (correct) next gate.
        # if stepped_through_gate:
        #     reward += 100
        # # Reward for reaching goal position (after navigating the gates in the correct order).
        # if at_goal_pos:
        #     reward += 100
        # # Penalize by collision.
        # if currently_collided:
        #     reward -= 1000
        # # Penalize by constraint violation.
        # if cnstr_violation:
        #     reward -= 100
        #####################################################################
        #####################################################################

        vicon_obs = [vicon.pos[0], 0, vicon.pos[1], 0, vicon.pos[2], 0, vicon.rpy[0], vicon.rpy[1], vicon.rpy[2], 0, 0, 0]
        command_type, args = ctrl.cmdFirmware(curr_time, vicon_obs, reward, done, info)

        # print(vicon.pos)
        # print(current_target_gate_id, gate_dist) 
        
        # ---- save the cmd for logging
        log_cmd.append([curr_time, rospy.get_time(), command_type, args]) 


        # Select interface.
        print("CMD", command_type, "ARGS", args)
        if command_type == Command.FULLSTATE:
            # print(args)
            # args = [args[0], [0,0,0], [0,0,0], 0, [0,0,0]]
            cf.cmdFullState(*args)
        elif command_type == Command.TAKEOFF:
            cf.takeoff(*args)
        elif command_type == Command.LAND:
            cf.land(*args)
        elif command_type == Command.STOP:
            cf.land(targetHeight=0, duration=2.0)
        elif command_type == Command.GOTO:
            cf.goTo(*args)
        elif command_type == Command.NOTIFYSETPOINTSTOP:
            cf.land(targetHeight=0, duration=2.0)
        elif command_type == Command.NONE:
            pass
        elif command_type == Command.FINISHED:
            break
        else:
            raise ValueError("[ERROR] Invalid command_type.")

        timeHelper.sleepForRate(CTRL_FREQ)

        if completed:
            break


    cf.land(0, 3)
    timeHelper.sleep(3.5)


    # ---- save the command as pkl
    # print(log_cmd)
    with open('../decode_pkl/cmd_test_arg_video1.pkl', 'wb') as f:
        pkl.dump(log_cmd,f)

    # print("press button to land...")
    # swarm.input.waitUntilButtonPressed()nsform.rotat
