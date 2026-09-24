import gymnasium as gym
import temporal_comp_gym
import pygame
import sys

def play():
    # Initialize the environment with human rendering and sequence dependencies
    env = gym.make('TemporalComp/HazardousDelivery-v0', render_mode='human', task_dependencies=True)
    obs, info = env.reset()
    
    # Render once to spawn the pygame window
    env.render()
    
    print("=======================================")
    print("    Hazardous Delivery - Human Play    ")
    print("=======================================")
    print(" Controls:")
    print("   [Arrow Keys] : Move Up, Down, Left, Right")
    print("   [ R ]        : Reset the environment manually")
    print("   [ ESC ]      : Quit")
    print("=======================================\n")

    running = True
    while running:
        # We manually process pygame events to capture key presses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
                
            if event.type == pygame.KEYDOWN:
                action = None
                
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_r:
                    print("Environment reset manually.")
                    env.reset()
                    env.render()
                    continue
                elif event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                
                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if reward != 0:
                        print(f"Reward Received: {reward}")
                        
                    env.render()
                    
                    if terminated or truncated:
                        print(f"Episode Finished (Terminated: {terminated}, Truncated: {truncated}). Resetting...\n")
                        env.reset()
                        env.render()

    env.close()
    pygame.quit()

if __name__ == "__main__":
    play()
