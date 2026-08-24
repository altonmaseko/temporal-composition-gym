import gymnasium as gym
import temporal_comp_gym
import time

# 1. Verification: Environment Creation & Agent Compatibility
env = gym.make('TemporalComp/SequentialColour-v0', render_mode='human')
obs, info = env.reset(seed=42)

print(f"Initial Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# 2. Verification: Step Rollout Loop & Visual Display
for step in range(600):
    # Sample a random discrete action (0=Up, 1=Down, 2=Left, 3=Right)
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    # Slow down loop slightly to watch the memorisation phase colours flash
    time.sleep(0.01)  
    
    if terminated or truncated:
        print(f"Episode finished at step {step} with reward {reward}. Resetting...")
        obs, info = env.reset()

env.close()
print("Sequential Colour Task verification complete!")