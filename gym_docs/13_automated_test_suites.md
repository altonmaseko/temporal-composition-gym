# Automated Test Suites and Verification Scripts

The `tests/` directory contains verification scripts for each of the three environments. Rather than utilizing strict unit testing frameworks (like `pytest` or `unittest`) to assert specific mathematical boundaries or isolated class behaviors, these scripts function as **Smoke Tests** and **Rollout Verifications**. 

Their primary purpose is to ensure that the environments strictly conform to the `Gymnasium` API standard and that their internal engines (Minigrid, Pymunk, Pygame) do not crash when subjected to rapid, unpredictable agent inputs.

---

## Core Verification Strategies

Across all three test files, the scripts systematically validate four core operational requirements before an RL agent is allowed to train on them:

### 1. Registration and Instantiation
They verify that the Python package architecture correctly exposes the environment string IDs to the global `gym` registry.
```python
env = gym.make('TemporalComp/DeferredMaintenance-v0', render_mode='human')
```

### 2. API Conformance
They verify that the core `reset()` and `step()` methods correctly unpack the required 5-tuple mandated by modern Gymnasium standards (as opposed to the older 4-tuple standard from deprecated OpenAI Gym versions).
```python
# Modern Gymnasium API signature validation
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

### 3. Space Validation
They explicitly print out the `observation_space` and `action_space` properties to allow developers to verify the tensor shapes and data types (e.g., `Box`, `Discrete`, `Dict`) before passing them to a neural network builder.

### 4. Random Rollout Stress Testing
Instead of manual human play, the scripts use `env.action_space.sample()` to bombard the environment with random valid actions. This ensures that the physics engines (e.g., AABB collision, rigid body overlap) and the game state loops (e.g., sequence transitions) handle erratic behavior gracefully without throwing exceptions or memory leaks.

---

## File Specific Implementations

### 1. Hazardous Delivery (`test_hazardous_delivery.py`)
This script performs a rapid 100-step rollout, randomly sampling discrete grid actions (0-3). It ensures that the Minigrid backend properly processes erratic random walks without breaking spatial boundaries or wrapper conversions.

```python
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.5)
```
> 📁 [`tests/test_hazardous_delivery.py`](../tests/test_hazardous_delivery.py)

### 2. Deferred Maintenance (`test_deferred_maintenance.py`)
This script runs a 300-step rollout testing the continuous physics engine (`pymunk`). It verifies that sampling continuous float arrays for thrust and torque does not break the simulation or crash the observation dictionary wrapper. It also includes an optional commented-out test line specifically designed to verify gravity resistance manually.

```python
for step in range(300):
    # Sample a random continuous action (thrust and rotation)
    action = env.action_space.sample() 
    
    # Optional: Hardcode a slight upward thrust to test gravity resistance
    # action = np.array([0.0, 1.0]) 
    
    obs, reward, terminated, truncated, info = env.step(action)
```
> 📁 [`tests/test_deferred_maintenance.py`](../tests/test_deferred_maintenance.py)

### 3. Sequential Colour (`test_sequential_colour.py`)
This script performs a longer 600-step rollout because it must verify stability across all three temporal phases (Memorisation, Retention, Navigation). It throttles the rendering loop to ensure the rapid RGB visual flashes are actually observable to the human tester monitoring the smoke test.

```python
for step in range(600):
    # Sample a random discrete action (0=Up, 1=Down, 2=Left, 3=Right)
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    # Slow down loop slightly to watch the memorisation phase colours flash
    time.sleep(0.01)  
```
> 📁 [`tests/test_sequential_colour.py`](../tests/test_sequential_colour.py)
