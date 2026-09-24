# Human Interaction and Testing Scripts

The `play_human/` directory contains three standalone Python scripts designed to allow developers (or human test subjects) to manually play and interact with each environment. These scripts demonstrate how to initialize the environments using Gymnasium's `render_mode='human'` and how to bridge Pygame's keyboard event listeners with the complex action spaces of the underlying simulation.

All scripts follow a standard game loop pattern: initialize, reset, poll events, map input to actions, step the environment, and render.

---

## 1. Hazardous Delivery (`play_hazardous_delivery.py`)

This script handles discrete grid movement. Because grid movement in Minigrid is absolute (one press equals one tile), the script relies on the Pygame `KEYDOWN` event queue, ensuring that holding a key doesn't cause the agent to zoom uncontrollably across the map in a single frame.

### Key Bindings
*   **Arrow Keys**: Map directly to discrete actions `0` (Up), `1` (Down), `2` (Left), and `3` (Right).
*   **[R]**: Allows the user to manually trigger an `env.reset()`.
*   **[ESC]**: Quits the game loop.

### Core Execution Loop
```python
while running:
    # Process pygame events one by one to prevent rapid-fire movement
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            action = None
            
            # Map physical keys to environment action integers
            if event.key == pygame.K_UP: action = 0
            elif event.key == pygame.K_DOWN: action = 1
            elif event.key == pygame.K_LEFT: action = 2
            elif event.key == pygame.K_RIGHT: action = 3
            elif event.key == pygame.K_r:
                env.reset()
                env.render()
                continue
            
            # Step the environment ONLY when a valid key is pressed
            if action is not None:
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
```
> 📁 [`play_human/play_hazardous_delivery.py`](../play_human/play_hazardous_delivery.py)

---

## 2. Deferred Maintenance (`play_deferred_maintenance.py`)

Unlike Hazardous Delivery, this environment operates in a continuous physics space (`pymunk`). Therefore, the script uses `pygame.key.get_pressed()` to poll the continuous *state* of the keys every frame, allowing the user to hold down keys to apply continuous physics forces like thrust and torque.

### Key Bindings
*   **W / UP**: Thrust (Y-axis = `1.0`)
*   **S / DOWN**: Air Brakes (Y-axis = `-1.0`)
*   **A / LEFT**: Rotate Counter-Clockwise (X-axis/Torque = `1.0`)
*   **D / RIGHT**: Rotate Clockwise (X-axis/Torque = `-1.0`)

### Core Execution Loop
```python
while running:
    # Poll continuous key states
    keys = pygame.key.get_pressed()
    
    thrust = 0.0
    torque = 0.0
    
    if keys[pygame.K_w] or keys[pygame.K_UP]: thrust = 1.0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]: thrust = -1.0
        
    if keys[pygame.K_a] or keys[pygame.K_LEFT]: torque = 1.0
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]: torque = -1.0
        
    # Construct continuous Box action tensor
    action = np.array([thrust, torque], dtype=np.float32)
    
    # Step the environment every frame
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```
> 📁 [`play_human/play_deferred_maintenance.py`](../play_human/play_deferred_maintenance.py)

---

## 3. Sequential Colour (`play_sequential_colour.py`)

This script is the most complex because it must handle the environment's three distinct temporal phases. The user only has control during the `navigation` phase; during the memory phases, the script must automatically step the environment forward so the sequence plays out automatically.

### Key Bindings
*   **WASD / Arrow Keys**: Map to discrete actions `0` (Up), `1` (Down), `2` (Left), and `3` (Right). Only active during Phase 3.

### Core Execution Loop
The script accesses `env.unwrapped` to read the internal `current_phase` variable. It aggressively throttles the Pygame clock to `5 FPS` during memory phases to make the colors flash at a human-readable pace (as 5 steps = 1 real-life second based on internal engine timing), then speeds back up to `30 FPS` for smooth gameplay during navigation.

```python
while running:
    core_env = env.unwrapped
    action = None
    
    if core_env.current_phase in ["memorisation", "retention"]:
        # Auto-advance time during memory phases.
        action = 0 
        clock.tick(5) # Throttle to 5 FPS 
    else:
        # Navigation phase: grant human control
        clock.tick(30) # Boost back to 30 FPS for smooth gameplay
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: action = 0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action = 1
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action = 2
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action = 3
            
    # Step the environment based on manual input OR auto-advancement
    if action is not None:
        obs, reward, terminated, truncated, info = env.step(action)
```
> 📁 [`play_human/play_sequential_colour.py`](../play_human/play_sequential_colour.py)
