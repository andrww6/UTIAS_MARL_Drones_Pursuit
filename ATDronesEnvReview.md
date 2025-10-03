# Environment.py Analysis: AT-Drones Execution Flow and Key Functions

## 1. Chronological Execution Flow

The AT-Drones environment follows a specific execution sequence that creates the simulation loop for the multi-drone pursuit scenario. Here's the chronological order of function calls when running the environment:

1. **Environment Creation** → `__init__(self, config)`
   - Environment parameters are set up
   - State variables are initialized
   - Physical dimensions and obstacle layouts are configured

2. **Environment Reset** → `reset(self)`
   - Agent and evader states are reset to starting conditions
   - Initial positions and orientations are set
   - `update_feature()` → Updates obstacle information
   - `update_state()` → Prepares initial observation state

3. **Main Loop (repeated for each timestep)**
   - `step(self, all_actions, mate_types, attention_score_array, ctrl_num)` 
     - `update_feature_last()` → Stores previous state
     - `transfer_action_to_final_F()` → Converts actions to forces
     - Compute evader movements via `escaper()` function
     - Update positions and orientations
     - `reward()` → Calculate rewards and check termination
     - `update_feature()` → Update obstacle information
     - `update_state()` → Prepare next observation state
     - Return new state, rewards, done flags, and info

4. **Episode Termination** (when any condition in `reward()` is met)
   - All evaders captured (success)
   - Time limit reached (timeout)
   - Collision detected (failure)

This cycle continues until the episode terminates, at which point `reset()` is called to begin a new episode.

## 2. Key Function Analysis

### 2.1 Environment Initialization

The `__init__` function sets up the entire environment structure, including physical parameters, state variables, and obstacle layouts:

```python
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
    # ... [more initializations] ...
    
    # Obstacle configuration based on env_idx
    self.env_idx = config["env_idx"]
    if self.env_idx == 0:
        # ... [obstacle definitions for environment 0] ...
    elif self.env_idx == 1:
        # ... [obstacle definitions for environment 1] ...
    elif self.env_idx == 2:
        # ... [obstacle definitions for environment 2] ...
```

Key points about initialization:
- The state space dimension is calculated based on the number of evaders and agents
- Physical parameters like velocities and time intervals are set from config
- Different obstacle layouts are defined based on the environment index

### 2.2 Step Function

The `step` function is the heart of the environment, advancing the simulation by one time step. It processes actions, calculates movements, and prepares the next state:

```python
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
```

Key aspects of the step function:

1. **Pursuer Movement Calculation**:
   - Each pursuer's action is converted to a force vector via `transfer_action_to_final_F`
   - The force is scaled by velocity and time interval to get position change
   - This applies to both controlled and teammate agents

2. **Evader Movement Calculation**:
   - For each uncaptured evader, a complex movement pattern is calculated
   - The `virtual_obstacle` buffer creates circular repulsion zones around other evaders
   - The `friends_position` variable holds positions of other evaders to avoid
   - These are passed to the `escaper` function which implements the evasion behaviors
   - The `escaper` function returns various state variables including zigzag counters and flags

3. **State Updates**:
   - Pursuer and evader positions are updated based on calculated forces
   - Orientation vectors are updated (equal to the normalized force vectors)
   - The `reward()` function calculates rewards and checks termination conditions
   - `update_feature()` and `update_state()` prepare the next observation state

### 2.3 Reward Function

The `reward` function calculates rewards for agents and determines when episodes terminate:

```python
def reward(self):
    reward = np.zeros((1, self.num_agent))  # reward buffer
    done = np.zeros((1, self.num_agent))  # done buffer
    capture_flag = False
    
    # Capture reward
    for j in range(self.num_evader):
        for i in range(self.num_agent):
            if np.linalg.norm(self.agent_position[:, i:i + 1] - self.target_position[:, j:j + 1]) < 300 and \
                    self.capture[0, j] == 0:
                capture_flag = True  # r_main
                self.capture[0, j] = 1
    if capture_flag:
        reward += 10
        
    # Termination conditions
    if np.all(self.capture):
        done[:] = 1  # Success
    elif self.t == 1000:
        done[:] = 2.  # Timeout
        
    for i in range(self.num_agent):
        # Collision penalty
        if np.linalg.norm(self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) < 100:
            # if the distance from the nearest obstacle is less than 100 mm
            reward[0, i] += self.neg_reward
            done[0, i] = 3  # Collision with obstacle
            
        # Additional proximity penalties (optional)
        if self.more_r:
            all_mate_distance = np.linalg.norm(self.agent_position[:, i:i + 1] - self.agent_position, axis=0)
            for index, mate_distance in enumerate(all_mate_distance):
                if mate_distance != 0:
                    if mate_distance > 200 and mate_distance < 251:
                        # very dangerous distance
                        reward[0, i] += self.neg_reward * 0.1
                    elif mate_distance > 250 and mate_distance < 301:
                        reward[0, i] += self.neg_reward * 0.01
                    elif mate_distance > 300 and mate_distance < 401:
                        reward[0, i] += self.neg_reward * 0.001

        # Agent collision penalty
        distance_nearest_teamate = np.amin(
            np.linalg.norm(self.agent_position[:, i:i + 1] - np.delete(self.agent_position, i, axis=1),
                         axis=0))
        if distance_nearest_teamate < 200:
            # if the distance from the nearest teammate exceeds 200 mm
            reward[0, i] += self.neg_reward
            done[0, i] = 3  # Collision with teammate
            
        # Distance-based reward
        for j in range(self.num_evader):
            if self.capture[0, j] == 0:
                distance_from_target_last = np.linalg.norm(
                    self.target_position_last[:, j:j + 1] - self.agent_position_last[:, i:i + 1])
                distance_from_target = np.linalg.norm(
                    self.target_position[:, j:j + 1] - self.agent_position[:, i:i + 1])
                reward[0, i] += (distance_from_target_last - distance_from_target) / 300  # Reward for getting closer

    # Propagate collision termination
    if np.any(done == 3):
        done[done == 0] = 2
        
    return reward, done
```

The reward function has several important components:

1. **Capture Reward (+10)**:
   - When any agent gets within 300mm of an evader, all agents receive +10 reward
   - The evader is marked as captured (`self.capture[0, j] = 1`)

2. **Termination Conditions**:
   - Success (`done = 1`): All evaders are captured
   - Timeout (`done = 2`): Time limit (1000 steps) is reached
   - Failure (`done = 3`): Collision with obstacle or other agent detected

3. **Collision Penalties**:
   - Obstacle collision: Applied when agent is within 100mm of obstacle
   - Agent collision: Applied when agents are within 200mm of each other
   - Both result in negative reward (`self.neg_reward`, default -10)

4. **Progressive Rewards**:
   - Distance reduction reward: Agents receive small rewards for decreasing distance to evaders
   - Optional proximity warnings: Graduated penalties for getting too close to other agents

5. **Collision Propagation**:
   - If any agent collides (`done = 3`), all other agents also terminate (`done = 2`)

### 2.4 Update Feature Function

The `update_feature` function is responsible for calculating the closest obstacles for each agent:

```python
def update_feature(self):
    virtual_obstacle = np.zeros((2, 0))  # virtual obstacle buffer
    for i in range(self.num_evader):
        if self.capture[0, i] == 1:  # if evader is captured
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
```

Key aspects of update_feature:

1. **Virtual Obstacle Creation**:
   - Captured evaders become circular virtual obstacles
   - These are represented as 36 points forming a circle with 100mm radius around the captured evader

2. **Nearest Obstacle Calculation**:
   - For environments with 3 agents: Finds the closest point from real obstacles and virtual obstacles
   - For environments with >3 agents: Also considers other agents as potential obstacles
   - This is used for collision detection and agent observation

The virtual obstacles are a critical part of the environment dynamics, as they:
- Prevent agents from colliding with captured evaders
- Force uncaptured evaders to avoid locations where other evaders were captured
- Create dynamic obstacles that change as the episode progresses

### 2.5 Update State Function

The `update_state` function calculates the observation vector for each agent:

```python
def update_state(self):
    self.state = np.zeros((self.num_state, self.num_agent))  # clear the environment state
    self.global_state = np.zeros((self.num_state, self.num_agent))  # global state

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

            # Update global state
            self.global_state[j * 2, i] = np.linalg.norm(temp) / 5000
            self.global_state[j * 2 + 1, i] = angle / np.pi

            # If evader is captured, mask it
            if self.capture[0, j] == 1:
                self.state[j * 2, i] = 2
                self.state[j * 2 + 1, i] = 0
                self.global_state[j * 2, i] = 2
                self.global_state[j * 2 + 1, i] = 0

        # Similar calculations for obstacles and other agents...
```

Key aspects of the update_state function:

1. **Observation Structure**:
   - For each agent, the state vector contains:
     - Distance and bearing to each evader (if visible)
     - Distance and bearing to nearest obstacle
     - Distance and bearing to each other agent (if visible)

2. **Perception Range Limitation**:
   - Entities outside perception range (`self.r_perception`, default 2000mm) are masked
   - Masked values use (2, 0) for distance and angle
   - Global state always includes all entities (used for visualization/debugging)

3. **Normalization**:
   - Distances are normalized by dividing by 5000 (environment height)
   - Angles are normalized by dividing by π

4. **Special Cases**:
   - Captured evaders are masked with (2, 0)
   - Cross product determines whether angle is positive or negative

### 2.6 Transfer Action to Final F Function

The `transfer_action_to_final_F` function translates agent actions into force vectors:

```python
def transfer_action_to_final_F(self, all_actions, attention_score_array, index, current_agent_type):
    if current_agent_type in ["PPO", "IPPO", "MAPPO", 'POAM']:
        c_action = np.clip(all_actions, -1, 1)
        agent_orientation_angle = np.arctan2(self.agent_orientation[1, :], self.agent_orientation[0, :]).reshape(1, self.num_agent)
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
    # Other agent types handling...
```

#### Key aspects of the transfer_action_to_final_F function:

1. **PPO Agent Translation**:
   - Actions are continuous values in [-1, 1]
   - Each action represents a steering angle change (±45 degrees maximum)
   - These are converted to a new orientation vector using trigonometry
   - The resulting force vector is normalized (unit length)

2. **DQN Agent Translation**:
   - Actions select parameters for Artificial Potential Field (APF) method
   - The `from_action_to_APF` function calculates repulsive and attractive forces
   - Maximum turning angle is limited to 45 degrees per step
   - If the desired turn exceeds this, the maximum allowed turn is used

3. **Common Features**:
   - Both methods produce unit vectors representing orientation
   - These are then scaled by velocity in the step function
   - This separation of orientation and velocity is a key simplification

#### Understanding the Force Vector's Role in AT-Drones

##### The `transfer_action_to_final_F` Function Explained

When examining the `transfer_action_to_final_F` function for PPO agents:

```python
def transfer_action_to_final_F(self, all_actions, attention_score_array, index, current_agent_type):
    if current_agent_type in ["PPO", "IPPO", "MAPPO", 'POAM']:
        c_action = np.clip(all_actions, -1, 1)
        agent_orientation_angle = np.arctan2(self.agent_orientation[1, :], self.agent_orientation[0, :]).reshape(1, self.num_agent)
        temp1 = np.cos(np.radians(c_action * 45) + agent_orientation_angle)
        temp2 = np.sin(np.radians(c_action * 45) + agent_orientation_angle)
        force = np.vstack((temp1, temp2))
        return force[:, index:index + 1]
```

This function takes the PPO agent's action (a value between -1 and 1) and converts it into a normalized force vector that represents the agent's new heading direction.

##### How the Force Vector is Used

Looking at the `step` function, the force vector `F` is used in TWO critical ways:

```python
def step(self, all_actions, mate_types, attention_score_array=np.array([0]), ctrl_num=None):
    # ...
    F = np.zeros((2, self.num_agent))
    agent_position_buffer = np.zeros((2, self.num_agent))
    for i in range(self.num_agent):
        # ... [agent type determined] ...
        F[:, i: i + 1] = self.transfer_action_to_final_F(all_actions, attention_score_array, i, agent_type)
        agent_position_buffer[:, i:i + 1] = F[:, i:i + 1] * self.v * self.delta_t

    # ... [evader calculations] ...

    # Update positions and orientations
    self.agent_position = self.agent_position + agent_position_buffer  # update pursuers' positions
    self.agent_orientation = F  # update pursuers' headings
```

##### 1. Position Update

```python
agent_position_buffer[:, i:i + 1] = F[:, i:i + 1] * self.v * self.delta_t
self.agent_position = self.agent_position + agent_position_buffer
```

The force vector is multiplied by velocity and time step to calculate how far the agent moves. This creates the actual physical movement in the environment.

When the agent takes an action, this code:
1. Converts the action to a direction change (±45° maximum)
2. Calculates a unit vector in that direction
3. Multiplies by speed and time to get distance moved
4. Updates the agent's position accordingly

##### 2. Orientation Update

```python
self.agent_orientation = F
```

The force vector becomes the agent's new orientation/heading. This is critical because:
- It determines what the agent can see (angles to other entities are relative to orientation)
- It affects future movements (new actions change direction relative to current orientation)
- It's used in the next step's calculations for turning angles

##### Why This Is Essential to the Environment

The `transfer_action_to_final_F` function is the link between the reinforcement learning agent's decision (the action value) and the physical motion in the environment. Without it:

1. The agent would have no way to control its movement direction
2. The orientation wouldn't update, meaning the agent's perspective wouldn't change
3. Position updates wouldn't reflect the agent's decisions

This is the **core of the control model** in AT-Drones - it implements the simplified physics where:
- Agents move at constant speed (self.v)
- They only control their direction (via the action value)
- Direction changes are limited to ±45° per step

##### Simplified Physics Visualization

Consider an agent currently facing east (orientation = [1, 0]) that takes action 0.5 (turning 22.5° to the left):

1. The `transfer_action_to_final_F` function:
   - Converts action 0.5 to angle 22.5°
   - Calculates new direction: cos(22.5°) = 0.92, sin(22.5°) = 0.38
   - Returns force vector [0.92, 0.38]

2. The `step` function:
   - Uses this vector to update position: 
     - position += [0.92, 0.38] * velocity * delta_time
   - Updates orientation to [0.92, 0.38]

In the next step, the action will turn relative to this new orientation, creating smooth turning behavior.


## 3. Environment Dynamics and Features

### 3.1 Evader Behavior

Evaders in the AT-Drones environment have sophisticated evasion strategies:

1. **Zigzagging**: Periodic lateral movements to avoid direct pursuit
   - Controlled by `zigzag_count` and `zigzag_flag` variables
   - Direction alternates every ~40 timesteps

2. **Wall Following**: When near obstacles, evaders follow the wall surface
   - Helps evaders navigate around obstacles without getting trapped
   - Tracked by `escaper_wall_following` flag

3. **Slip Maneuvers**: Emergency evasion when pursuers get too close
   - Perpendicular movement to slip past pursuers
   - Tracked by `escaper_slip_flag`

4. **Virtual Obstacles**: Evaders treat captured evaders as obstacles
   - This creates dynamic obstacle patterns as the episode progresses
   - Captured evaders become circular regions to avoid

The `escaper` function combines these behaviors into a unified movement strategy, but its internal details aren't fully visible in the environment.py file.

### 3.2 Physical Constants

The environment uses several important physical constants:

1. **Pursuer Velocity**: `self.v` (default 300 mm/s)
2. **Evader Velocity**: `self.e_v` (default 600 mm/s, twice as fast as pursuers)
3. **Time Interval**: `self.delta_t` (default 0.1s)
4. **Perception Range**: `self.r_perception` (default 2000mm)
5. **Capture Threshold**: 300mm (distance to register a capture)
6. **Obstacle Collision Threshold**: 100mm
7. **Agent Collision Threshold**: 200mm

These constants define the physical dynamics of the environment and the challenge level for pursuing agents.

## 4. Summary

The AT-Drones environment implements a complex multi-agent pursuit-evasion scenario with the following key features:

1. **Multiple Agent Types**: Supports PPO, DQN, and other agent types with different action spaces
2. **Sophisticated Evader Behavior**: Implements zigzagging, wall following, and slip maneuvers
3. **Dynamic Obstacles**: Captured evaders become obstacles, creating evolving environment challenges
4. **Partial Observability**: Agents can only perceive entities within a limited range
5. **Collaborative Pursuit**: Rewards are shared among all pursuers when any evader is captured

The environment handles the full simulation cycle, from converting agent actions to physical movements, calculating rewards, updating environment state, and preparing observations for the next decision step. The simulation continues until either all evaders are captured, a collision occurs, or the time limit is reached.

The simplification of fixed velocity with orientation control makes the environment more tractable for reinforcement learning while still capturing the essential dynamics of drone pursuit scenarios.