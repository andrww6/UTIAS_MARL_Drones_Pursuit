# Andrew Yin – Summer Research 2025

This repository documents the development and testing of sim-to-real control for evader behavior from the MARL Drone Pursuit Project on Crazyflie drones. The project focuses on adapting simulation-based evader logic with physical execution using the Crazyswarm framework and Vicon tracking system.

---

## System Overview

- **Hardware**: Crazyflie 2.0, Vicon Motion Capture System (Myhal), FSC laptops
- **Software**: ROS1 Noetic, crazyswarm-import
- **Simulation-to-Real Objective**: Port rule-based and policy-driven drone behaviors into real world with stable, loggable drone flight
- **Github Link**: Link to project Github with all teammates' branches and code. https://github.com/0xC000005/MARL_Drones_Pursuit.git

---

## Directory Structure

| Folder | Description |
|--------|-------------|
| `evader_integration_test 1/` | Initial controller built on top of `edit_this.py` from AER1217; connects o_multi_layout rule-based behaviors (e.g. zigzag, Bézier) to Crazyflie. Includes modified `cmdFullStateFirmware`. |
| `evader-integration test 2/` | Final working implementation:  
  - `edit_this_custom1.py`: First-gen fully custom PID controller, initial sim-to-real translation test with broken logging system.
  - `edit_this_custom2.py`: Current and most stable controller with consistent logging and drone command execution, there are issues with the drone idling.
  - `offline_validation.py`: Allows simulated physical controller testing (approximate, not fully accurate).  
  - Includes dependencies: `base/`, `o_base/`, `o_multi_layout`, `utils/`, `crazyflie/`.|
| `idling_flightlogs/` | Logs from tests where drones received correct commands but remained idle — issue still work in progress. Demonstrates working logging system with real flight videos included|
| `straight_line_flightlogs/` | Manual testing with user walking drone to test logging accuracy. Propellers removed. |
| `Code Review/` | Internal methodology documentation and commentary on evader/pursuer logic, including state-space, reward-space, etc. definitions. |
| `HMARL Lit Review/` | Literature papers that influenced project design. |
| `Presentations/` | June monthly sprint slides and final summer research presentation. |
| `admin/`, `Readings/`, `reference/` | Organizational and reference material |

---

## Analysis, Project Status, Suggested Next Steps

Currently, edit_this_custom2.py is a working controller that correctly transmits commands generated from o_multi_layout to the drones; it also has a fully working log system. However, there may be an issue with
the rate at which it generates commands for the drone. The entire flight plan (all the commands) are generated in around 1 second, and the drone idling situation may be because crazyflies remain idle if they're
unable to safely reach the desired coordinates. ctrl_freq, the variable that controls the frequency at which commands are generated, is set to 30 right now--30 commands per second. Tweaking the variable does not
change the command frequency, indicating there is something wrong with it or frequency is actually defined somewhere else other than the controller file. I suggest looking into this issue to first ensure ctrl_freq
is working, and then work from there to troubleshoot the idling issue by lowering the frequency at which commands are generated.

---

## How to Run

> All scripts assume `crazyswarm-import` is installed and that the system is running under ROS1 noetic in ubuntu 20.04, simply use fsc laptops if possible--all of this was built on those systems.  
> Scripts are set up for Vicon tracking in MY580 by default.

