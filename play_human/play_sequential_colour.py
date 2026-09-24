import gymnasium as gym
import pygame
import time
import sys
import os

# Add the root directory to sys.path so we can import temporal_comp_gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import temporal_comp_gym.envs 

def play():
    print("Starting Sequential Colour Navigation Task (Human Mode)")
    print("Controls: Arrow Keys or W/A/S/D to move.")
    print("Close the window to exit.")
    
    # Instantiate the environment with human-friendly settings
    env = gym.make(
        'TemporalComp/SequentialColour-v0', 
        render_mode='human',
        sequence_length_range=(3, 5),
        agent_wall_distance_blocks=6,
        reward_density='sparse'
    )
    
    obs, info = env.reset()
    
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Pump pygame events to keep window responsive and handle close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Access the underlying environment to check the current phase
        core_env = env.unwrapped
        
        action = None
        
        if core_env.current_phase in ["memorisation", "retention"]:
            # Auto-advance time during memory phases.
            # The core env flashes colours every 5 steps.
            # We throttle to 5 FPS here so 5 steps = 1 second per colour flash.
            action = 0 
            clock.tick(5)
        else:
            # Navigation phase
            # Throttle to 30 FPS for smooth gameplay
            clock.tick(30)
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                action = 0
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                action = 1
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                action = 2
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                action = 3
                
        # Only step the environment if an action is taking place
        # (Auto-steps during memory, manual steps during navigation)
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode ended! Final Score: {core_env.score:.1f}")
                time.sleep(1.5)
                obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    play()
