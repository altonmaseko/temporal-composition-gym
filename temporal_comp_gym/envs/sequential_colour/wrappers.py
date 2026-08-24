import gymnasium as gym
import numpy as np

class GaussianNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, observation_noise_level=0.0):
        super().__init__(env)
        self.noise_level = observation_noise_level
        
    def observation(self, obs):
        if self.noise_level > 0.0:
            noise = np.random.normal(0, self.noise_level * 255, obs.shape)
            noisy_obs = np.clip(obs + noise, 0, 255).astype(np.uint8)
            return noisy_obs
        return obs
