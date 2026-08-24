import gymnasium as gym
import temporal_comp_gym
import time

env = gym.make('TemporalComp/HazardousDelivery-v0', render_mode='human')
obs, info = env.reset(seed=42)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.5)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()