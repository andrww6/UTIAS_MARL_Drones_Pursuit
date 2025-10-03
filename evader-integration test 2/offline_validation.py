import numpy as np
import time
import matplotlib.pyplot as plt
from edit_this import Controller  # Uses your existing Controller class

def run_offline_validation(duration=10, ctrl_freq=1):
    # Fake initial info, similar to real firmware
    initial_info = {
        "ctrl_timestep": 1.0 / ctrl_freq,
        "ctrl_freq": ctrl_freq,
        "quadrotor_kf": None
    }
    initial_obs = np.zeros(6)

    controller = Controller(initial_obs, initial_info, use_firmware=True, verbose=True)

    # Fake drone state (simulate physics)
    pos = np.zeros(3)
    vel = np.zeros(3)
    dt = 1.0 / ctrl_freq

    all_target, all_actual = [], []

    for t in range(int(duration * ctrl_freq)):
        #time.sleep(0.1)  # TODO: CHECK THE FREQUENCY
        time_sec = t * dt

        # Pretend we get obs from the drone
        obs = np.zeros(6)
        obs[0], obs[2], obs[4] = pos

        # Instead of calling cmdFirmware, we directly step the evader behavior
        if not hasattr(controller, "evader_initialized"):
            controller.cmdFirmware(time_sec, obs)  # First call initializes behavior
            continue

        controller.behavior.step()
        target_pos = controller.behavior.goal_positions[0].copy()
        target_vel = controller.behavior.goal_velocities[0].copy()
        #print(f"[SIM DEBUG] t={time_sec:.2f}s, Goal Pos {target_pos}, Goal Vel {target_vel}")
        #print(controller.behavior.goal_positions)
        # Clamp velocities to simulate firmware safety
        vel_norm = np.linalg.norm(target_vel)
        if vel_norm > 0.8:
            target_vel *= 0.8 / vel_norm

        # Simulate drone movement (simple Euler physics)
        vel = target_vel
        pos += vel * dt

        # Log for later analysis
        all_target.append(target_pos.copy())
        all_actual.append(pos.copy())
  

    return np.array(all_target), np.array(all_actual)

def plot_results(target_arr, actual_arr):
    plt.figure(figsize=(8, 6))
    plt.plot(target_arr[:, 0], target_arr[:, 1], 'b--', label='Target Path')
    plt.plot(actual_arr[:, 0], actual_arr[:, 1], 'r-', label='Simulated Actual Path')
    for t, a in zip(target_arr, actual_arr):
        plt.plot([t[0], a[0]], [t[1], a[1]], 'k-', alpha=0.3)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Offline Validation: Target vs Simulated Actual Path")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("offline_trajectory.png")
    plt.show()

if __name__ == "__main__":
    target, actual = run_offline_validation(duration=300, ctrl_freq=1)
    plot_results(target, actual)
    print("[DONE] Check 'offline_trajectory.png' for results.")

