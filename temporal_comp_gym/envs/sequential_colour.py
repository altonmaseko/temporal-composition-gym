import gymnasium as gym
import numpy as np

class SequentialColourEnv(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        # TODO: Define self.observation_space and self.action_space
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # TODO: Initialize environment state
        # return observation, info
        return None, {}
        
    def step(self, action):
        # TODO: Implement transition logic based on action
        # return observation, reward, terminated, truncated, info
        return None, 0.0, False, False, {}
