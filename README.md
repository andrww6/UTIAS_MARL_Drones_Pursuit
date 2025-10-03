# Multi-Agent Drone Pursuit — Selected Works (UTIAS Summer Research)

This repository is a **selection of work** from my Summer 2025 research internship at the **University of Toronto Institute for Aerospace Studies (UTIAS), Flight Systems & Control Lab**.  
It highlights the controllers, training utilities, and experiment logs I worked on while contributing to a **multi-drone pursuit–evasion environment**.

## Contents
- **Controllers**
  - Custom Python PID controllers for autonomous drone pursuit.
  - Designed for integration with the Crazyflie platform and simulator.
- **Logging System**
  - Python logging framework for tracking planned vs. actual drone flight paths.
  - Includes performance metrics (capture time, path deviation, coordination efficiency).
- **Test Flight Logs**
  - Raw flight logs and recorded test data from physical and simulated experiments.
- **Training Files**
  - A small excerpt of training scripts and bash launchers used to train reinforcement learning policies locally and on Compute Canada clusters.

## Project Background
The project focused on **reinforcement learning–based policy development** for drone swarms in pursuit–evasion tasks.  

## Stack
- **Languages & Libraries:** Python, PyTorch, TensorFlow, NumPy, pandas, Matplotlib  
- **Platforms:** ROS1, Crazyflie, Compute Canada HPC clusters  
- **Methods:** PID control, policy iteration, reinforcement learning, swarm coordination  
- **Github Link**: Link to original project Github with all teammates' branches and code. https://github.com/0xC000005/MARL_Drones_Pursuit.git

## Directory Structure

| Folder | Description |
|--------|-------------|
| `evader_integration_test 1/` | Initial controller built on top of `edit_this.py` from AER1217; connects o_multi_layout rule-based behaviors (e.g. zigzag, Bézier) to Crazyflie. Includes modified `cmdFullStateFirmware`. |
| `evader-integration test 2/` | Final working implementation:  
  - `edit_this_custom1.py`: First-gen fully custom PID controller, initial sim-to-real translation test with non-functional logging system.
  - `edit_this_custom2.py`: Current and most stable controller with consistent logging and drone command execution, there are issues with the drone idling.
  - `offline_validation.py`: Allows simulated physical controller testing (approximate, not fully accurate).  
  - Includes dependencies: `base/`, `o_base/`, `o_multi_layout`, `utils/`, `crazyflie/`.|
| `idling_flightlogs/` | Logs from tests where drones received correct commands but remained idle — issue still work in progress. Demonstrates working logging system with real flight videos included|
| `straight_line_flightlogs/` | Manual testing with user walking drone to test logging accuracy. Propellers removed. |
| `Code Review/` | Internal methodology documentation and commentary on evader/pursuer logic, including state-space, reward-space, etc. definitions. |
| `HMARL Lit Review/` | Literature papers that influenced project design. |
| `Presentations/` | June monthly sprint slides and final summer research presentation. |
| `admin/`, `Readings/`, `reference/` | Organizational and reference material |
