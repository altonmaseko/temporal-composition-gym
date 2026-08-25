import gymnasium as gym
import pygame
import numpy as np
import temporal_comp_gym

def main():
    print("Initializing Deferred Maintenance Task for manual testing...")
    env = gym.make('TemporalComp/DeferredMaintenance-v0', render_mode='human')
    obs, info = env.reset(seed=42)
    
    print("\n" + "="*50)
    print("MANUAL CONTROL ENABLED")
    print("="*50)
    print("Controls:")
    print("  [W] or [UP ARROW]    : Apply Thrust (Combat Gravity)")
    print("  [S] or [DOWN ARROW]  : Air Brakes (Sudden Stop)")
    print("  [A] or [LEFT ARROW]  : Rotate Left (Counter-Clockwise)")
    print("  [D] or [RIGHT ARROW] : Rotate Right (Clockwise)")
    print("Close the Pygame window to exit.")
    print("="*50 + "\n")
    
    running = True
    total_reward = 0.0
    
    while running:
        # Pump the Pygame event queue to prevent OS freezes and capture input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Read keyboard state
        keys = pygame.key.get_pressed()
        
        # Action space: [thrust (Y-axis), angular momentum / torque (X-axis)]
        thrust = 0.0
        torque = 0.0
        
        # Upward Thrust / Air Brakes
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            thrust = 1.0
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            thrust = -1.0
            
        # Angular Rotation
        # PyMunk uses standard math conventions: Positive torque = Counter-Clockwise (Left)
        # Negative torque = Clockwise (Right)
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            torque = 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            torque = -1.0
            
        # Construct the continuous action array
        action = np.array([thrust, torque], dtype=np.float32)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # The core environment handles the 60fps clock ticking in its render method
        env.render()
        
        if terminated or truncated:
            print(f"🏁 Episode finished! Total Reward: {total_reward:.2f}. Resetting...")
            obs, info = env.reset()
            total_reward = 0.0

    print("Closing environment...")
    env.close()

if __name__ == "__main__":
    main()
