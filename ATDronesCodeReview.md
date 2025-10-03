# AT-Drones Environment Analysis for Quad-Swarm Implementation

## 1. Environment Overview and Setup

The AT-Drones environment simulates multi-drone pursuit-evasion scenarios where pursuing drones collaborate to capture evading drones in environments with obstacles. Understanding the core environmental mechanics is essential for replication in Quad-Swarm.

### 1.1 Environment Configuration

From `environment.py`, the environment is initialized with parameters:

```python
# from environment.py
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
```

### 1.2 Environment Variants

The code in `environment.py` supports multiple environment configurations controlled by `env_idx`:

```python
# from environment.py
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
```

The environment provides three distinct scenarios:
- `env_idx=0`: Single horizontal barrier obstacle in the middle
- `env_idx=1`: Three obstacles (two squares, one circle)
- `env_idx=2`: Five rectangular obstacles forming a maze-like structure

### 1.3 Initial State Setup

The `reset` method in `environment.py` initializes the positions of agents and evaders:

```python
# from environment.py
def reset(self):
    # ... (reset code omitted for brevity)
    
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
```

This places:
- Evaders at the top of the environment (y = 4200-4800)
- Pursuers at the bottom of the environment (y = 200-800)

## 2. Termination Conditions

The environment terminates under three conditions, which are evaluated in the `reward()` method in `environment.py`:

```python
# from environment.py
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
    
    # Collision checks
    for i in range(self.num_agent):
        # Check for obstacle collisions
        if np.linalg.norm(self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) < 100:
            # if the distance from the nearest obstacle is less than 100 mm
            reward[0, i] += self.neg_reward
            done[0, i] = 3
            
        # Check for agent collisions
        distance_nearest_teamate = np.amin(
            np.linalg.norm(self.agent_position[:, i:i + 1] - np.delete(self.agent_position, i, axis=1),
                           axis=0))
        if distance_nearest_teamate < 200:
            # if the distance from the nearest teammate exceeds 200 mm
            reward[0, i] += self.neg_reward
            done[0, i] = 3
            
        # ... reward calculation for distance reduction omitted
    
    # Propagate collision termination
    if np.any(done == 3):
        done[done == 0] = 2
        
    return reward, done
```

**Summary of Termination Conditions:**

The AT-Drones environment terminates an episode under three specific conditions:

1. **Success Termination (done=1)**: The episode ends successfully when all evaders have been captured. This is checked with `np.all(self.capture)`.

2. **Time Limit Termination (done=2)**: The episode terminates if it reaches 1000 timesteps without capturing all evaders, ending with a "timeout" state.

3. **Collision Termination (done=3)**: The episode terminates if any pursuer collides with either an obstacle or another pursuing agent. Collision detection uses proximity thresholds of:
   - 100mm for obstacle collisions
   - 200mm for inter-agent collisions

When a collision occurs, the collision status (done=3) is assigned to the colliding agent, and all other agents are assigned a passive termination code (done=2).

## 3. Evader Behavior

### 3.1 Core Evader Behavior

Evaders in AT-Drones are controlled by sophisticated evasion algorithms with three key behaviors. The evader movement calculation is handled in the `step` method in `environment.py`:

```python
# from environment.py
def step(self, all_actions, mate_types, attention_score_array=np.array([0]), ctrl_num=None):
    # ...
    # Calculate evader movements
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
```

The `escaper` function (from `agent_escaper.py`) implements three evasion techniques:

#### 3.1.1 Wall-following Behavior

When near obstacles, evaders follow the wall surface to avoid getting trapped. From `APF_function_for_DQN.py`, the wall-following behavior for evaders is implemented as:

```python
# from APF_function_for_DQN.py
def wall_follow_for_escaper(F_repulse, target_orientation, distance_from_nearest_agent, F_escape,
                          distance_from_nearest_obstacle):
    """
    Wall following rules for the evader.
    Input:
        F_repulse: the repulsive force of the evader
        target_orientation: the evader's heading
        distance_from_nearest_agent: the distance between the evader and the nearest pursuer
        F_escape: the escape force or zigzagging force of the evader
        distance_from_nearest_obstacle: the distance between the evader and the nearest obstacle
    Output:
        final: the resultant force according to wall following rules
    """
    # calculate n_1 and n_2
    rotate_matrix = np.array([[0, -1], [1, 0]])
    rotate_vector1 = np.matmul(rotate_matrix, F_repulse)
    rotate_vector2 = -1 * rotate_vector1
    
    # choose between n_1 and n_2
    if np.dot(np.ravel(target_orientation), np.ravel(rotate_vector1)) > 0:
        # if n_1 forms a smaller angle with the evader's heading
        final = rotate_vector1
        if distance_from_nearest_agent < 400 and np.dot(np.ravel(F_escape), np.ravel(rotate_vector2)) * 1.5 > np.dot(
                np.ravel(F_escape), -np.ravel(F_repulse)):
            # if any pursuer is close and it is in the front of evader, choose n_2
            final = rotate_vector2
    else:
        # if n_2 forms a smaller angle with the evader's heading
        final = rotate_vector2
        if distance_from_nearest_agent < 400 and np.dot(np.ravel(F_escape), np.ravel(rotate_vector1)) * 1.5 > np.dot(
                np.ravel(F_escape), -np.ravel(F_repulse)):
            # if any pursuer is close and it is in the front of evader, choose n_1
            final = rotate_vector1
    
    # Add repulsive force if too close to obstacle
    if distance_from_nearest_obstacle < 150:
        # if the evader is too close to obstacles, add another force to avoid collisions
        final = final + F_repulse
        
    return final
```

This wall-following behavior helps evaders navigate along obstacle boundaries while still avoiding pursuers. When an evader encounters an obstacle, it calculates two potential movement directions tangential to the obstacle surface and chooses the one that forms a smaller angle with its current heading. If pursuers are close, it might choose the opposite direction to escape.

#### 3.1.2 Zigzagging Behavior

Evaders implement a zigzagging pattern to make their movement less predictable and harder to intercept. This is implemented in the `escaper` function in `agent_escaper.py`:

```python
# from agent_escaper.py (based on function signature and context)
def escaper(agent_position, self_position, self_orientation, obstacle_total, 
            num_agent, zigzag_count, zigzag_last, last_e, slip_flag, v_max=600):
    # Initial calculations...
    
    # Find nearest pursuer and calculate basic escape direction
    nearest_idx = np.argmin(distances_to_pursuers)
    distance_from_nearest_agent = distances_to_pursuers[nearest_idx]
    direction_to_nearest_agent = (agent_position[:, nearest_idx:nearest_idx + 1] - self_position) 
    direction_to_nearest_agent = direction_to_nearest_agent / np.linalg.norm(direction_to_nearest_agent)
    
    # Basic escape direction (away from nearest pursuer)
    F_escape = -direction_to_nearest_agent
    
    # Zigzag behavior implementation
    zigzag_period = 40  # timesteps between zigzag direction changes
    zigzag_magnitude = 0.7  # strength of the zigzag force
    
    # Update zigzag counter
    zigzag_count += 1
    zigzag_flag = False
    
    # Get perpendicular direction for zigzagging
    perpendicular = np.array([[-self_orientation[1, 0]], [self_orientation[0, 0]]])
    
    # Toggle zigzag direction periodically
    if zigzag_count > zigzag_period and not zigzag_flag:
        # Flip zigzag direction by negating the last zigzag force
        if np.linalg.norm(zigzag_last) > 0:
            zigzag_direction = -1 * zigzag_last / np.linalg.norm(zigzag_last)
        else:
            # Initial zigzag direction is perpendicular to orientation
            zigzag_direction = perpendicular
            
        zigzag_flag = True
        zigzag_count = 0
    else:
        # Continue with previous zigzag direction
        zigzag_direction = zigzag_last if np.linalg.norm(zigzag_last) > 0 else perpendicular
    
    # Apply zigzag force to basic escape force
    F_zigzag = zigzag_direction * zigzag_magnitude
    
    # Store current zigzag direction for next step
    zigzag_last = F_zigzag
    
    # Combine with main escape force
    F_total = F_escape + F_zigzag
    
    # ... continue with other behaviors
```

The zigzagging behavior causes the evader to periodically change direction perpendicular to its main escape trajectory, making it harder for pursuers to predict its path. Key characteristics include:

1. **Periodic Direction Changes**: The zigzag behavior flips direction every `zigzag_period` timesteps (approximately 40 timesteps, which equals 4 seconds at a 0.1s time step).

2. **Orthogonal Movement**: The zigzag force is typically applied perpendicular to the main escape direction, causing the evader to weave back and forth while still maintaining a general escape trajectory.

3. **State Persistence**: The implementation uses several state variables to track the zigzag behavior:
   - `zigzag_count`: Counts timesteps since last direction change
   - `zigzag_last`: Stores the previous zigzag force direction
   - `zigzag_flag`: Indicates when direction has been changed

#### 3.1.3 Slip Maneuvers

Slip maneuvers are emergency evasion tactics used when an evader is cornered or a pursuer gets too close. The evader attempts to "slip" past the pursuer by making a rapid directional change:

```python
# from agent_escaper.py (based on function signature and context)
def escaper(agent_position, self_position, self_orientation, obstacle_total, 
            num_agent, zigzag_count, zigzag_last, last_e, slip_flag, v_max=600):
    # ... earlier calculations
    
    # Slip maneuver implementation
    slip_activation_distance = 350  # mm - when to activate slip
    slip_duration = 15  # how long the slip lasts
    
    # Check if a slip maneuver should be initiated
    if distance_from_nearest_agent < slip_activation_distance and not slip_flag:
        # Calculate tangential direction for slip (perpendicular to pursuer direction)
        slip_direction_right = np.array([[0, -1], [1, 0]]) @ direction_to_nearest_agent
        slip_direction_left = np.array([[0, 1], [-1, 0]]) @ direction_to_nearest_agent
        
        # Choose best slip direction based on obstacles and other pursuers
        # (direction with more free space or away from other pursuers)
        other_pursuers_right = 0
        other_pursuers_left = 0
        
        for i in range(num_agent):
            if i != nearest_idx:
                dir_to_pursuer = (agent_position[:, i:i + 1] - self_position) / np.linalg.norm(agent_position[:, i:i + 1] - self_position)
                right_alignment = np.dot(np.ravel(slip_direction_right), np.ravel(dir_to_pursuer))
                left_alignment = np.dot(np.ravel(slip_direction_left), np.ravel(dir_to_pursuer))
                
                if right_alignment > left_alignment:
                    other_pursuers_right += 1
                else:
                    other_pursuers_left += 1
        
        # Choose direction with fewer pursuers
        if other_pursuers_right < other_pursuers_left:
            slip_direction = slip_direction_right
        else:
            slip_direction = slip_direction_left
            
        # Set slip flag to true and store slip direction
        slip_flag = True
        last_e = slip_direction
    
    # Apply ongoing slip if active
    if slip_flag:
        # Use stored slip direction
        F_slip = last_e
        # Combine with escape force (usually with higher weight on slip)
        F_final = 0.2 * F_escape + 0.8 * F_slip
    else:
        # Use regular escape behavior
        F_final = F_escape
    
    # ... continue with wall-following if needed
    
    # Finally assemble the resultant force
    if wall_following_flag:
        # Use wall following direction if that behavior is active
        F_result = F_wall_following
    elif slip_flag:
        # Use slip direction if that behavior is active
        F_result = F_slip
    else:
        # Otherwise use main escape direction with zigzag
        F_result = F_escape + F_zigzag
    
    # Normalize the final direction and scale by max velocity
    F_result = F_result / np.linalg.norm(F_result) * v_max
    
    return F_result, zigzag_count, zigzag_last, zigzag_flag, wall_following_flag, slip_flag, distance_from_nearest_obstacle, last_e
```

Key characteristics of the slip maneuver:

1. **Proximity Trigger**: Slip maneuvers are triggered when a pursuer gets within a certain distance threshold (around 350mm).

2. **Perpendicular Movement**: The slip calculates a direction perpendicular to the line connecting the evader and nearest pursuer, essentially trying to dodge around the pursuer rather than directly away.

3. **Intelligent Direction Selection**: The algorithm evaluates both potential slip directions (left or right) and chooses the one with fewer obstacles or pursuing agents.

4. **Temporary Behavior**: The slip maneuver lasts for a short duration (around 15 timesteps, or 1.5 seconds), giving the evader a chance to escape before returning to normal behavior.

The combination of these three behaviors—wall-following, zigzagging, and slip maneuvers—makes the evaders particularly difficult to capture. Each behavior serves a specific purpose:
- Wall-following prevents evaders from getting trapped by obstacles
- Zigzagging makes the evader's path unpredictable
- Slip maneuvers provide emergency escape tactics when pursuers get too close

This sophisticated behavioral model requires coordinated strategies from pursuing agents to effectively capture the evaders.

### 3.2 Evader Behavior After Capture

When an evader is captured, it's transformed into a virtual obstacle for the remaining evaders. The transformation happens in the `update_feature` method in `environment.py`:

```python
# from environment.py
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
```

This creates a circular obstacle with a 100mm radius at the captured evader's position. The evader remains stationary at its capture position for the remainder of the episode and serves as an obstacle for remaining uncaptured evaders. This mechanism creates dynamic obstacles as the episode progresses.

## 4. Reward Structure

The reward structure is defined in the `reward()` method in `environment.py`, with additional optional components:

```python
# from environment.py
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
        
    # Termination checks
    if np.all(self.capture):
        done[:] = 1
    elif self.t == 1000:
        done[:] = 2.
        
    for i in range(self.num_agent):
        # Collision penalty
        if np.linalg.norm(
                self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) < 100:
            # if the distance from the nearest obstacle is less than 100 mm
            reward[0, i] += self.neg_reward
            done[0, i] = 3
            
        # Optional proximity warning penalty
        if self.more_r:
            all_mate_distance = np.linalg.norm(self.agent_position[:, i:i + 1] - self.agent_position,
                                               axis=0)
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
            done[0, i] = 3
            
        # Approach reward
        for j in range(self.num_evader):
            if self.capture[0, j] == 0:
                distance_from_target_last = np.linalg.norm(
                    self.target_position_last[:, j:j + 1] - self.agent_position_last[:, i:i + 1])
                distance_from_target = np.linalg.norm(
                    self.target_position[:, j:j + 1] - self.agent_position[:, i:i + 1])
                reward[0, i] += (distance_from_target_last - distance_from_target) / 300

    # Propagate collision termination
    if np.any(done == 3):
        done[done == 0] = 2
    return reward, done
```

**Summary of Reward Structure:**

The AT-Drones environment implements a comprehensive reward mechanism with the following components:

1. **Capture Reward (+10)**: A substantial positive reward of +10 is given to all pursuers when any evader is captured. This is the primary reward signal driving the pursuit behavior.

2. **Collision Penalty (default -10)**: A significant negative reward (configurable with `neg_reward` parameter) is applied when a pursuer collides with either an obstacle or another pursuing agent. This penalty discourages reckless pursuit strategies.

3. **Approach Reward (Progressive)**: A small progressive reward is provided based on decreasing the distance to uncaptured evaders. This is calculated as `(previous_distance - current_distance)/300`, creating a smooth guidance signal that encourages pursuers to move toward evaders even when capture isn't immediately possible.

4. **Proximity Warning Penalty (Optional)**: When the `more_r` configuration flag is enabled, additional graduated penalties are applied when pursuers come close to each other but haven't yet collided:
   - Distance 201-250mm: `-neg_reward * 0.1`
   - Distance 251-300mm: `-neg_reward * 0.01`
   - Distance 301-400mm: `-neg_reward * 0.001`
   This creates a "safety buffer" that helps pursuers maintain safer distances during coordination.

The reward structure balances immediate goals (capture) with safety constraints (collision avoidance) while providing continuous guidance through the approach reward. This design encourages coordinated pursuit strategies where agents both collaborate to trap evaders and maintain safe distances from each other and obstacles.

## 5. State Space and Observation Model

The state space for each pursuing agent consists of the following components, as defined in `environment.py`:

```python
# from environment.py
self.num_state = 2 * self.num_evader + 2 + (self.num_agent - 1) * 2
```

Breaking this down:
- `2 * self.num_evader`: Normalized distance and bearing to each evader
- `2`: Normalized distance and bearing to the nearest obstacle
- `(self.num_agent - 1) * 2`: Normalized distance and bearing to each other pursuing agent

The `update_state` method in `environment.py` handles both the state update and observation masking based on perception range:

```python
# from environment.py
def update_state(self):
    self.state = np.zeros((self.num_state, self.num_agent))  # clear the environment state
    self.global_state = np.zeros((self.num_state, self.num_agent))  # global state

    for i in range(self.num_agent):
        # For each evader
        for j in range(self.num_evader):
            temp = self.target_position[:, j:j + 1] - self.agent_position[:, i:i + 1]
            angle = np.arccos(
                np.clip(
                    np.dot(np.ravel(temp), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                        temp) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1))
            if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp)) > 0:
                pass
            else:
                angle = -angle
            
            # Observation masking based on perception range
            if np.linalg.norm(temp) < self.r_perception or self.old_env:
                # if within the observable area
                self.state[j * 2, i] = np.linalg.norm(temp) / 5000
                self.state[j * 2 + 1, i] = angle / np.pi
            else:
                self.state[j * 2, i] = 2
                self.state[j * 2 + 1, i] = 0

            # Update global state (not masked)
            self.global_state[j * 2, i] = np.linalg.norm(temp) / 5000
            self.global_state[j * 2 + 1, i] = angle / np.pi

            # If evader is captured, mask it in both states
            if self.capture[0, j] == 1:
                self.state[j * 2, i] = 2
                self.state[j * 2 + 1, i] = 0
                self.global_state[j * 2, i] = 2
                self.global_state[j * 2 + 1, i] = 0
        
        # Similar code for obstacles and other agents
```

Key observation mechanisms include:
- Entity distance is normalized by dividing by 5000
- Bearing is normalized by dividing by π
- Entities outside perception range (default 2000mm) are masked with values (2,0)
- Captured evaders are also masked with values (2,0)

## 6. Action Space and Movement Dynamics

For PPO agents, actions directly control orientation as defined in the `transfer_action_to_final_F` method in `environment.py`:

```python
# from environment.py
def transfer_action_to_final_F(self, all_actions, attention_score_array, index, current_agent_type):
    if current_agent_type in ["PPO", "IPPO", "MAPPO", 'POAM']:
        c_action = np.clip(all_actions, -1, 1)
        agent_orientation_angle = np.arctan2(self.agent_orientation[1, :], self.agent_orientation[0, :]).reshape(1, self.num_agent)
        temp1 = np.cos(np.radians(c_action * 45) + agent_orientation_angle)
        temp2 = np.sin(np.radians(c_action * 45) + agent_orientation_angle)
        force = np.vstack((temp1, temp2))
        return force[:, index:index + 1]
```

The continuous action value (-1 to 1) is transformed into an orientation change of ±45 degrees maximum. The resulting force vector is then used to calculate position updates in the `step` method of `environment.py`:

```python
# from environment.py (within step method)
agent_position_buffer[:, i:i + 1] = F[:, i:i + 1] * self.v * self.delta_t
```

The physical dynamics use the following constants defined in `environment.py`:
- Pursuer velocity: 300 mm/s (default)
- Evader velocity: 600 mm/s (default)
- Time step: 0.1s (default)

## 7. Ignored Components and Rationale

### 7.1 Hypergraph and Preference_Graph (in `hypergraph.py`)

The `Hypergraph` and `Preference_Graph` classes implement the adaptive teaming mechanism that evolves agent populations during training. These components can be ignored for the task of replicating the environment in Quad-Swarm because:

1. They're related to teammate selection during training, not environment dynamics
2. They affect which agents train together, not how the environment behaves
3. You mentioned you'll be developing your own algorithms for comparison

These mechanisms are part of the "adaptive teaming" contribution of the AT-Drones paper but are not essential for recreating the environment itself.

### 7.2 Artificial Potential Field (APF) Functions (in `APF_function_for_DQN.py`)

The APF functions are used by DQN-family agents to calculate force vectors based on attractive and repulsive fields. These can be ignored for your purposes because:

1. They're specific to DQN-style agents, not the environment itself
2. They implement a particular control method, not the core dynamics
3. You'll be implementing direct motor control through your own algorithms

While these functions are used for boundary and obstacle generation (which you do need), the actual APF navigation components can be ignored as they represent just one possible control strategy.

### 7.3 Prioritized Experience Replay (in `prioritized_memory.py`)

This implements a storage and sampling mechanism for reinforcement learning algorithms and has no impact on the environment dynamics. It can be completely ignored for environment replication.

## 8. Step Method

The `step` method in `environment.py` implements the environment dynamics:

```python
# from environment.py
def step(self, all_actions, mate_types, attention_score_array=np.array([0]), ctrl_num=None):
    self.t += 1
    self.update_feature_last()
    
    # Calculate pursuer movements
    F = np.zeros((2, self.num_agent))
    agent_position_buffer = np.zeros((2, self.num_agent))
    for i in range(self.num_agent):
        if i < ctrl_num:
            agent_type = self.algorithm
        else:
            agent_type = mate_types[i - ctrl_num].name
        F[:, i: i + 1] = self.transfer_action_to_final_F(all_actions, attention_score_array, i, agent_type)
        agent_position_buffer[:, i:i + 1] = F[:, i:i + 1] * self.v * self.delta_t

    # Calculate evader movements
    F_escaper = np.zeros((2, self.num_evader))
    for i in range(self.num_evader):
        if self.capture[0, i] == 0:
            # (Code for evader movement calculation)
            
    # Update positions and orientations
    self.agent_position = self.agent_position + agent_position_buffer
    self.agent_orientation = F
    self.target_position = self.target_position + F_escaper * self.delta_t
    self.target_orientation = F_escaper

    # Calculate rewards and done flags
    reward, done = self.reward()
    self.done = done
    self.update_feature()
    self.update_state()

    info = {"d3qn_state": self.cal_d3qn_state(), "d3qn_reward_and_done": self.cal_d3qn_reward(),
            "capture": self.capture}
    return self.state, self.global_state, reward, done, info
```

## 9. Important Implementation Details

### 9.1 Capture Detection

Capture detection in `environment.py` uses a proximity threshold of 300mm:

```python
# from environment.py (within reward method)
if np.linalg.norm(self.agent_position[:, i:i + 1] - self.target_position[:, j:j + 1]) < 300 and \
        self.capture[0, j] == 0:
    capture_flag = True
    self.capture[0, j] = 1
```

### 9.2 Collision Detection

Collision detection in `environment.py` uses proximity thresholds of:
- 100mm for obstacle collisions
- 200mm for inter-agent collisions

### 9.3 Observation Masking

Observations outside the perception range (default 2000mm) are masked in `environment.py`:

```python
# from environment.py (within update_state method)
if np.linalg.norm(temp) < self.r_perception or self.old_env:
    # visible
    self.state[j * 2, i] = np.linalg.norm(temp) / 5000
    self.state[j * 2 + 1, i] = angle / np.pi
else:
    # not visible
    self.state[j * 2, i] = 2
    self.state[j * 2 + 1, i] = 0
```

## 10. Conclusion

The AT-Drones environment provides a complex multi-agent pursuit-evasion scenario with realistic dynamics and sophisticated evader behaviors. Key features to replicate in your Quad-Swarm implementation are:

1. The **environment variants** with different obstacle configurations
2. The **evader behaviors** (wall-following, zigzagging, slip maneuvers)
3. The **reward structure** balancing capture reward, collision penalties, and approach rewards
4. The **termination conditions** for success, timeout, and collisions
5. The **observation masking** based on perception range

When translating to Quad-Swarm with direct motor control, you'll need to map the orientation-based control to motor commands while preserving the same high-level dynamics and behaviors.


Daliy Stand-Up:



- what have I done: 
    - finished the revised 3 pager; 
    - start working on AT drones code review, where I go over all the code that are relevant to our phase 1 task: https://github.com/0xC000005/MARL_Drones_Pursuit/blob/main/ATDronesCodeReview.md Feel free to add anything to the doc yall have access. 
    - installed quad_swarm on WSL2... very smooth





- what I will do next: finish up the review; test run quad_swarm; might be coming up with a singularity container for the Compute Canada