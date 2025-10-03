import numpy as np
import time as time_pkg
import matplotlib.pyplot as plt
from collections import deque
import atexit #crash still logging handling
from project_utils import Command, PIDController
from o_multi_layout import Scenario_o_multi_layout

DURATION = 20
MAX_DEVIATION_ALLOWED = 1e9  # Adjust if needed

class Controller:
    def __init__(self, initial_obs, initial_info, use_firmware: bool = False, buffer_size: int = 100, verbose: bool = False):
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        if use_firmware:
            self.ctrl = None
        else:
            self.ctrl = PIDController()
            self.KF = initial_info["quadrotor_kf"]

        self.reset()
        self.interEpisodeReset()

        self.target = []
        self.actual = []
        self._duration = DURATION
        self.i_offset = 0

        with open("log.txt", "w") as f:
            pass

    def cmdFirmware(self, time, obs, reward=None, done=None, info=None):
        # time_pkg.sleep(0.1)  # TODO: CHECK THE FREQUENCY
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        iteration = int(time * self.CTRL_FREQ) - self.i_offset

        if not hasattr(self, 'evader_initialized'):
            class DummyEnv:
                def __init__(self):
                    self.control_freq = 30.0
                    self.tick = 0
                    self.ep_time = 0.0
                    self.dynamics = type("dynamics", (), {
                        "pos": np.zeros(3),
                        "vel": np.zeros(3),
                        "acc": np.zeros(3),
                        "omega": np.zeros(3),
                        "rot": np.eye(3),
                    })()

            dummy_envs = [DummyEnv()]
            num_agents = 1
            atexit.register(self._finalize_logging)
            # ONCE I CHANGED THE ROOM DIMS, STARTED WORKING PROPERLLY
            room_dims = [10, 10, 3]
            obst_pos_arr = np.array([[0, 0, 0]])
            obst_size = 0.5

            self.behavior = Scenario_o_multi_layout(
                #quads_mode='static_same_goal',
                quads_mode='o_multi_layout',
                envs=dummy_envs,
                num_agents=num_agents,
                room_dims=room_dims
            )
            self.behavior.reset(
                obst_map=None,
                cell_centers=None,
                obst_pos_arr=obst_pos_arr,
                obst_size=obst_size
            )
            self.evader_initialized = True
            print("Evader scenario initialized")

        curr_pos = np.array([obs[0], obs[2], obs[4]])

        if iteration == 0:
            with open("log.txt", "a") as f:
                f.write("# x_target,y_target,z_target,deviation,x_actual,y_actual,z_actual\n")
            return Command(2), [1.0, 2]  # Takeoff

        elif 3 * self.CTRL_FREQ <= iteration < (self._duration + 3) * self.CTRL_FREQ:
            self.behavior.step()
            target_pos = self.behavior.goal_positions[0].copy()
            target_vel = self.behavior.goal_velocities[0].copy()

            vel_norm = np.linalg.norm(target_vel)
            if vel_norm > 0.8:
                target_vel *= (0.8 / vel_norm)

            deviation = obs[:6:2] - target_pos
            if np.linalg.norm(deviation) > MAX_DEVIATION_ALLOWED:
                self.i_offset += 1

            self.target.append(target_pos)
            self.actual.append(obs[:6:2])
            with open("log.txt", "a") as f:
                f.write(f"{target_pos[0]},{target_pos[1]},{target_pos[2]},{np.linalg.norm(deviation)},{obs[0]},{obs[2]},{obs[4]}\n")

            return Command(1), [target_pos, target_vel, np.zeros(3), 0.0, np.zeros(3)]

        elif iteration == int((self._duration + 3.5) * self.CTRL_FREQ):
            return Command(6), []  # Notify setpoint stop

        elif iteration == int((self._duration + 3.6) * self.CTRL_FREQ):
            return Command(3), [0.0, 1]  # Land

        elif iteration == int((self._duration + 10) * self.CTRL_FREQ):
            target_arr = np.array(self.target)
            actual_arr = np.array(self.actual)
            print(target_arr, actual_arr)
            plt.figure(figsize=(8, 6))
            plt.plot(target_arr[:, 0], target_arr[:, 1], 'b--', label='Target Path')
            plt.plot(actual_arr[:, 0], actual_arr[:, 1], 'r-', label='Actual Path')

            for t, a in zip(target_arr, actual_arr):
                plt.plot([t[0], a[0]], [t[1], a[1]], 'k-', alpha=0.3)

            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title("XY Trajectory: Target vs Actual with Correspondence")
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("trajectory_correspondence.png")

            return Command(4), []  # Stop

        return Command(0), []  # None

    def reset(self):
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.interstep_counter = 0
        self.interepisode_counter = 0

    def interEpisodeReset(self):
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
    def _finalize_logging(self):

        if self.target and self.actual:
            target_arr = np.array(self.target)
            actual_arr = np.array(self.actual)

            plt.figure(figsize=(8, 6))
            plt.plot(target_arr[:, 0], target_arr[:, 1], 'b--', label='Target Path')
            plt.plot(actual_arr[:, 0], actual_arr[:, 1], 'r-', label='Actual Path')

            for t, a in zip(target_arr, actual_arr):
                plt.plot([t[0], a[0]], [t[1], a[1]], 'k-', alpha=0.3)

            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title("XY Trajectory: Target vs Actual with Correspondence")
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("trajectory_correspondence.png")
            print("[LOG] Saved trajectory_correspondence.png")

        else:
            print("[LOG] No data to plot.")
    
        
