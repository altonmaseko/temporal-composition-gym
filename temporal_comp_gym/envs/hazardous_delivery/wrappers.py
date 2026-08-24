import gymnasium as gym
import numpy as np

class HazardousDeliveryObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, include_health_in_state=True, include_inventory_in_state=True):
        super().__init__(env)
        self.include_health = include_health_in_state
        self.include_inventory = include_inventory_in_state
        
        # 2 for pos, 4 for surroundings, 1 for current square
        obs_dim = 2 + 4 + 1
        if self.include_health:
            obs_dim += 1
        if self.include_inventory:
            obs_dim += 1
            
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def observation(self, obs):
        env = self.env.unwrapped
        x, y = env.agent_pos
        
        def get_type_val(obj):
            if obj is None: return 0
            if obj.type == 'damage': return 1
            if obj.type == 'lava': return 2
            if obj.type in ['package', 'destination']: return 3
            if obj.type == 'goal': return 4
            return 0 
            
        grid = env.grid
        up = grid.get(x, y-1)
        down = grid.get(x, y+1)
        left = grid.get(x-1, y)
        right = grid.get(x+1, y)
        curr = grid.get(x, y)
        
        surroundings = [
            get_type_val(up),
            get_type_val(down),
            get_type_val(left),
            get_type_val(right)
        ]
        
        curr_val = get_type_val(curr)
        vec = [float(x), float(y)] + surroundings + [float(curr_val)]
        
        if self.include_health:
            vec.append(float(env.health))
        if self.include_inventory:
            # Using delivered packages as proxy for inventory memory challenge
            vec.append(float(env.delivered_packages))
            
        return np.array(vec, dtype=np.float32)
