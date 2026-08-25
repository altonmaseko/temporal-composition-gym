import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DeferredMaintenanceObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Determine the size of the micro-navigation sensors
        spatial_sensor_type = self.unwrapped.spatial_sensor_type
        if spatial_sensor_type == 'local_grid':
            micro_nav_size = 9 # 3x3 local binary grid
        else:
            micro_nav_size = 8 # 8 simulated ray-cast lines
            
        # Determine the size of station radar (X and Y per color)
        self.num_colors = self.unwrapped.max_code_types
        radar_size = self.num_colors * 2
        
        # Kinematics(6) + MicroNav(9/8) + MacroNav(2) + FloorCode(1) + Readiness(1) + Radar
        obs_size = 6 + micro_nav_size + 2 + 1 + 1 + radar_size
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
    def observation(self, obs):
        # We assume the core env returns a dictionary with these components
        kinematics = np.array([
            obs['pos_x'], obs['pos_y'],
            obs['vel_x'], obs['vel_y'],
            obs['angle'], obs['ang_vel']
        ], dtype=np.float32)
        
        micro_nav = np.array(obs['micro_nav'], dtype=np.float32).flatten()
        macro_nav = np.array(obs['macro_nav'], dtype=np.float32).flatten()
        
        floor_code = np.array([obs['floor_code']], dtype=np.float32)
        readiness = np.array([obs['readiness']], dtype=np.float32)
        
        station_radar = np.array(obs['station_radar'], dtype=np.float32).flatten()
        
        flat_obs = np.concatenate([
            kinematics, 
            micro_nav, 
            macro_nav, 
            floor_code, 
            readiness, 
            station_radar
        ])
        
        return flat_obs
