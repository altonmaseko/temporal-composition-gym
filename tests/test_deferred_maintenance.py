import gymnasium as gym
import temporal_comp_gym
import time
import numpy as np

# 1. Verification: Environment Creation & Agent Compatibility
env = gym.make('TemporalComp/DeferredMaintenance-v0', render_mode='human')
obs, info = env.reset(seed=42)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# 2. Verification: Step Rollout Loop & Visual Display
for step in range(300):
    # Sample a random continuous action (thrust and rotation)
    action = env.action_space.sample() 
    
    # Optional: Hardcode a slight upward thrust to test gravity resistance
    # action = np.array([0.0, 1.0]) 
    
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.05)
    
    if terminated or truncated:
        print(f"Episode finished at step {step} with reward {reward}. Resetting...")
        obs, info = env.reset()

env.close()
print("Deferred Maintenance Task verification complete!")