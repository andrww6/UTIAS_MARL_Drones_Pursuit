import numpy as np
import random
from o_base import Scenario_o_base


class Scenario_o_multi_layout(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.room_dims = room_dims
        self.obstacle_layout = None
        self.goal_positions = []  # Static goal positions
        self.goal_velocities = []  # Goal velocities
        self.min_goal_distance = 1.5  # changed from 4.0 due to testing area constraints  # Minimum distance between goals
        self.goal_speed = 2.0  # Increased speed of goal movement
        self.goal_safe_distance = 1.0  # changed for my580 Safe distance between goals
        self.zigzag_distance = 1.3 #2.0  # Distance to travel before changing direction
        self.goal_distances_traveled = []  # Track distance traveled for each goal
        self.goal_last_positions = []  # Track last position for distance calculation
        self.goal_frozen = []  # Track if goals are frozen
        self.goal_capture_distance = 1.0  # Distance at which a goal is considered captured
        self.evaders = 4  # Number of evader drones 
        self.goal_landing_speed = 0.5  # Speed at which goals descend when captured
        self.goal_landing = []  # Track if goals are in landing phase

    def check_goal_separation(self, goal_pos, existing_goals, min_distance):
        """Check if a goal position is far enough from all existing goals"""
        for existing_goal in existing_goals:
            if np.linalg.norm(goal_pos - existing_goal) < min_distance:
                return False
        return True

    def check_goal_collision(self, pos1, pos2):
        """Check if two goals are too close to each other"""
        return np.linalg.norm(pos1 - pos2) < self.goal_safe_distance

    def check_obstacle_collision(self, pos):
        """Check if a position collides with any obstacle"""
        if not hasattr(self, 'obstacles') or self.obstacles is None:
            return False
            
        for obstacle_pos in self.obst_pos_arr:
            # Check if position is within obstacle bounds
            # Obstacles are 2x2 or 3x3 in size
            if len(self.obst_pos_arr) == 1:  # Layout 0: 3x3 obstacle
                size = 1.5 * self.obst_size
            else:  # Layout 1 or 2: 2x2 obstacles
                size = self.obst_size
                
            if (abs(pos[0] - obstacle_pos[0]) < size and 
                abs(pos[1] - obstacle_pos[1]) < size):
                return True
        return False

    def get_avoidance_direction(self, pos, current_vel):
        """Calculate a new direction to avoid obstacles"""
        # Try different angles to find a clear path
        angles = [45, 90, 135, 180, 225, 270, 315]  # Try these angles in degrees
        best_angle = None
        max_distance = 0
        
        for angle_deg in angles:
            angle_rad = angle_deg * (np.pi / 180)
            # Create rotation matrix
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1]
            ])
            
            # Try the new direction
            new_vel = rotation_matrix @ current_vel
            new_pos = pos + new_vel * 0.1  # Look ahead a bit
            
            # Check if this direction is clear
            if not self.check_obstacle_collision(new_pos):
                # Calculate distance to nearest obstacle
                min_dist = float('inf')
                for obstacle_pos in self.obst_pos_arr:
                    dist = np.linalg.norm(new_pos[:2] - obstacle_pos[:2])
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_distance:
                    max_distance = min_dist
                    best_angle = angle_rad
        
        if best_angle is None:
            # If no good angle found, reverse direction
            return -current_vel
            
        # Create rotation matrix for best angle
        cos_angle = np.cos(best_angle)
        sin_angle = np.sin(best_angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        
        # Return new velocity
        new_vel = rotation_matrix @ current_vel
        return new_vel / np.linalg.norm(new_vel) * self.goal_speed

    def reset(self, obst_map=None, cell_centers=None, obst_pos_arr=None, obst_size=None):
        # Randomly select a scenario layout
        self.obstacle_layout = 1  # Fixed layout
        self.obstacle_list = []
        room_x, room_y, room_z = self.room_dims
        

        # Generate static goal positions
        num_goals = 4  # Number of unique goals to generate
        self.goal_positions = []
        self.goal_velocities = []
        self.goal_distances_traveled = []
        self.goal_last_positions = []
        self.goal_frozen = []
        self.goal_landing = []
        self.obst_pos_arr = obst_pos_arr
        self.obst_size = obst_size
        margin = 2.0  # Margin from walls
        z_height = 1.0  # Fixed height for goals
        
        for i in range(num_goals):
            max_attempts = 100
            for attempt in range(max_attempts):
                new_goal_pos = np.array([
                    np.random.uniform(-room_x/2 + margin, room_x/2 - margin),
                    np.random.uniform(-room_y/2 + margin, room_y/2 - margin),
                    z_height  # Fixed z height
                ])
                
                # Check distance from other goals
                if self.check_goal_separation(new_goal_pos, self.goal_positions, self.min_goal_distance):
                    self.goal_positions.append(new_goal_pos)
                    # Initialize with random direction (only in x-y plane)
                    angle = np.random.uniform(0, 2 * np.pi)
                    velocity = np.array([np.cos(angle), np.sin(angle), 0]) * self.goal_speed
                    self.goal_velocities.append(velocity)
                    self.goal_distances_traveled.append(0.0)
                    self.goal_last_positions.append(new_goal_pos.copy())
                    self.goal_frozen.append(False)
                    self.goal_landing.append(False)
                    break
                
                if attempt == max_attempts - 1:
                    # If we can't find a good position, use the last generated one
                    new_goal_pos = np.clip(new_goal_pos, 
                                          [-room_x/2 + margin, -room_y/2 + margin, z_height],
                                          [room_x/2 - margin, room_y/2 - margin, z_height])
                    self.goal_positions.append(new_goal_pos)
                    # Initialize with random direction (only in x-y plane)
                    angle = np.random.uniform(0, 2 * np.pi)
                    velocity = np.array([np.cos(angle), np.sin(angle), 0]) * self.goal_speed
                    self.goal_velocities.append(velocity)
                    self.goal_distances_traveled.append(0.0)
                    self.goal_last_positions.append(new_goal_pos.copy())
                    self.goal_frozen.append(False)
                    self.goal_landing.append(False)

        # Initialize goals array for all environments
        self.goals = np.zeros((len(self.envs), 3))  # Initialize with zeros for all environments
        
        # Assign goals to drones (all except last 4 share goals, last 4 have no goals)
        for i in range(len(self.envs)):
            if i < self.num_agents - 4:  # All drones except last 4 share the 4 goals
                self.goals[i] = self.goal_positions[i % 4]  # Cycle through the 4 goals
            # Last 4 drones have no goals (zeros)
        
        # Set spawn points for all quadrotors
        self.spawn_points = []
        margin = 1.0  # Minimum distance from walls
        min_spawn_distance = 2.0  # Minimum distance between spawn points
        
        for i in range(len(self.envs)):
            if i >= self.num_agents - 4:  # Last 4 quadrotors
                goal_idx = i - (self.num_agents - 4)  # Get corresponding goal index (0-3)
                if goal_idx < len(self.goal_positions):  # Make sure we have a valid goal
                    self.spawn_points.append(self.goal_positions[goal_idx])  # Use the goal position as spawn point
                else:
                    # Fallback to random position if no valid goal
                    spawn_pos = np.array([
                        np.random.uniform(-room_x/2 + margin, room_x/2 - margin),
                        np.random.uniform(-room_y/2 + margin, room_y/2 - margin),
                        np.random.uniform(margin, room_z - margin)
                    ])
                    self.spawn_points.append(spawn_pos)
            else:
                # For other quadrotors, use random positions with minimum distance between them
                max_attempts = 100
                for attempt in range(max_attempts):
                    spawn_pos = np.array([
                        np.random.uniform(-room_x/2 + margin, room_x/2 - margin),
                        np.random.uniform(-room_y/2 + margin, room_y/2 - margin),
                        np.random.uniform(margin, room_z - margin)
                    ])
                    
                    # Check distance from other spawn points
                    too_close = False
                    for existing_spawn in self.spawn_points:
                        if np.linalg.norm(spawn_pos - existing_spawn) < min_spawn_distance:
                            too_close = True
                            break
                    
                    if not too_close:
                        break
                    
                    if attempt == max_attempts - 1:
                        # If we can't find a good position, use the last generated one
                        # but ensure it's within boundaries
                        spawn_pos = np.clip(spawn_pos, 
                                          [-room_x/2 + margin, -room_y/2 + margin, margin],
                                          [room_x/2 - margin, room_y/2 - margin, room_z - margin])
                
                self.spawn_points.append(spawn_pos)
        
        # Call super().reset() or other logic as needed for your environment.
        super().reset(obst_map=obst_map, cell_centers=cell_centers)

    def get_obstacle_list(self):
        return self.obstacle_list 

    def change_goal_direction(self, goal_idx):
        """Change the direction of a goal by a random angle between 120 and 240 degrees"""
        current_vel = self.goal_velocities[goal_idx]
        
        # Calculate random angle between 120 and 240 degrees
        angle = np.random.uniform(120, 240) * (np.pi / 180)  # Convert to radians
        
        # Create rotation matrix
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to velocity vector
        new_vel = rotation_matrix @ current_vel
        
        # Normalize and scale to maintain constant speed
        new_vel = new_vel / np.linalg.norm(new_vel) * self.goal_speed
        self.goal_velocities[goal_idx] = new_vel

    def step(self):
        dt = 1.0 / self.envs[0].control_freq
        
        # Update goal positions
        room_x, room_y, room_z = self.room_dims
        margin = 2.0
        
        #new_pos = self.goal_velocities[0].copy()
        
        # First update positions for non-frozen goals
        for i in range(len(self.goal_positions)):
        
            if not self.goal_frozen[i]:
                # Store current position before update
                old_pos = self.goal_positions[i].copy()
                
                # Check for obstacle collision before moving
                
                
                new_pos = self.goal_positions[i].copy()
                new_pos[:2] += self.goal_velocities[i][:2] * dt
                
                print(self.goal_positions)
             
                
                # Check for collisions with obstacles using obstacle map
                if hasattr(self, 'obst_map') and self.obst_map is not None:
                    # Convert position to grid coordinates
                    grid_size = 0.5  # This should match the grid_size used in obst_generation_given_density
                    grid_x = int((new_pos[0] + room_x/2) / grid_size)
                    grid_y = int((new_pos[1] + room_y/2) / grid_size)
                    
                    # Check if position is within bounds and collides with obstacle
                    if (0 <= grid_x < self.obst_map.shape[0] and 
                        0 <= grid_y < self.obst_map.shape[1] and 
                        self.obst_map[grid_x, grid_y] == 1):
                        # Change direction to avoid obstacle
                        self.goal_velocities[i] = self.get_avoidance_direction(
                            self.goal_positions[i], self.goal_velocities[i])
                        self.goal_distances_traveled[i] = 0.0
                        continue
                
                # Also check for collisions with obstacles using obstacle positions
                if hasattr(self, 'obst_pos_arr') and self.obst_pos_arr is not None:
                    collision = False
                    for obst_pos in self.obst_pos_arr:
                        # Check if position is within obstacle bounds
                        if (abs(new_pos[0] - obst_pos[0]) < self.obst_size + 0.5 and 
                            abs(new_pos[1] - obst_pos[1]) < self.obst_size + 0.5):
                            collision = True
                            break
                    
                    if collision:
                        # Change direction to avoid obstacle
                        self.goal_velocities[i] = self.get_avoidance_direction(
                            self.goal_positions[i], self.goal_velocities[i])
                        self.goal_distances_traveled[i] = 0.0
                        continue
                
                # If no collisions, update position (only x and y components)
                self.goal_positions[i][:2] = new_pos[:2]
                
                
               
                # Update distance traveled (only in x-y plane)
                distance_moved = np.linalg.norm(self.goal_positions[i][:2] - old_pos[:2])
                self.goal_distances_traveled[i] += distance_moved
          
             
                # Check if we need to change direction
                if self.goal_distances_traveled[i] >= self.zigzag_distance:
                    self.change_goal_direction(i)
                    self.goal_distances_traveled[i] = 0.0
               
                # Check room boundaries and reverse direction if needed
                if self.goal_positions[i][0] - margin < -room_x/2 or self.goal_positions[i][0] + margin > room_x/2:
                    self.goal_velocities[i][0] *= -1
                    self.goal_distances_traveled[i] = 0.0
                if self.goal_positions[i][1] - margin < -room_y/2 or self.goal_positions[i][1] + margin > room_y/2:
                    self.goal_velocities[i][1] *= -1
                    self.goal_distances_traveled[i] = 0.0
                
                # Ensure goals stay within boundaries
                self.goal_positions[i][:2] = np.clip(self.goal_positions[i][:2],
                                                   [-room_x/2 + margin, -room_y/2 + margin],
                                                   [room_x/2 - margin, room_y/2 - margin])
                                                   
        
            elif self.goal_landing[i]:  # If goal is in landing phase
                # Gradually decrease z coordinate
                self.goal_positions[i][2] = max(0.0, self.goal_positions[i][2] - self.goal_landing_speed * dt)
                if self.goal_positions[i][2] <= 0.0:  # If landed
                    self.goal_landing[i] = False  # Stop landing phase
                    self.goal_positions[i][2] = 0.0  # Ensure exactly at ground level
        
        # Check for goal captures by regular drones
        for i in range(len(self.goal_positions)):
            if not self.goal_frozen[i]:
                for j in range(self.num_agents - self.evaders):  # Only check regular drones
                    drone_pos = self.envs[j].dynamics.pos
                    distance = np.linalg.norm(drone_pos[:2] - self.goal_positions[i][:2])
                    if distance < self.goal_capture_distance:
                        # Start landing phase
                        self.goal_frozen[i] = True
                        self.goal_landing[i] = True
                        self.goal_velocities[i] = np.zeros(3)  # Stop movement
                        break
        
        
        # Then check for goal-goal collisions and resolve them
        for i in range(len(self.goal_positions)):
            for j in range(i + 1, len(self.goal_positions)):
                if not (self.goal_frozen[i] or self.goal_frozen[j]):  # Only check non-frozen goals
                    if self.check_goal_collision(self.goal_positions[i], self.goal_positions[j]):
                        # Move goals apart if they're too close
                        direction = self.goal_positions[i] - self.goal_positions[j]
                        distance = np.linalg.norm(direction)
                        if distance > 0:
                            direction = direction / distance
                            overlap = self.goal_safe_distance - distance
                            if overlap > 0:
                                # Move goals apart by half the overlap distance each
                                adjustment = direction * overlap * 0.5
                                self.goal_positions[i] += adjustment
                                self.goal_positions[j] -= adjustment
                                
                                # Ensure goals stay within room boundaries after adjustment (THIS WAS CAUSING THE BUG)
                                self.goal_positions[i][:2] = np.clip(self.goal_positions[i][:2],
                                                                   [-room_x/2 + margin, -room_y/2 + margin],
                                                                   [room_x/2 - margin, room_y/2 - margin])
                                self.goal_positions[j][:2] = np.clip(self.goal_positions[j][:2],
                                                                   [-room_x/2 + margin, -room_y/2 + margin],
                   
                                                                   [room_x/2 - margin, room_y/2 - margin])
        
        # Update goals for all environments
        for i in range(len(self.envs)):
            if i < self.num_agents - 4:  # All drones except last 4
                goal_idx = i % len(self.goal_positions)
                self.envs[i].goal = self.goal_positions[goal_idx]
                self.goals[i] = self.goal_positions[goal_idx]
            else:  # Last 4 drones
                goal_idx = i - (self.num_agents - 4)  # Get corresponding goal index (0-3)
                if goal_idx < len(self.goal_positions):  # Make sure we have a valid goal
                    # Set their positions to match the goals
                    self.envs[i].dynamics.pos = self.goal_positions[goal_idx].copy()
                    self.envs[i].dynamics.vel = np.zeros(3)  # Keep them stationary
                    self.envs[i].dynamics.acc = np.zeros(3)
                    self.envs[i].dynamics.omega = np.zeros(3)
                    self.envs[i].dynamics.rot = np.eye(3)  # Keep them upright
                self.goals[i] = np.zeros(3)  # No goals for these drones 
                
                
                
