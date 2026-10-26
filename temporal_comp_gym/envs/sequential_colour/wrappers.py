import gymnasium as gym
import numpy as np


class GaussianNoiseWrapper(gym.ObservationWrapper):
    """Adds pixel noise from the wrapper's own seeded generator.

    The generator is independent of the environment's RNG, so adding noise never changes
    sequences or room layouts. It is reseeded whenever reset() receives a seed.
    """

    def __init__(self, env, observation_noise_level=0.0):
        super().__init__(env)
        self.noise_level = observation_noise_level
        self._noise_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._noise_rng = np.random.default_rng([int(seed), 0x5EED])
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        if self.noise_level > 0.0:
            noise = self._noise_rng.normal(0, self.noise_level * 255, obs.shape)
            noisy_obs = np.clip(obs + noise, 0, 255).astype(np.uint8)
            return noisy_obs
        return obs
