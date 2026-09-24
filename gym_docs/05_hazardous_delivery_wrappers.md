# Hazardous Delivery Environment: Agent Observation Space and Wrappers

By default, the `minigrid` framework provides agents with a 3D ego-centric visual matrix (an image of what the agent sees in front of it). However, the Hazardous Delivery environment is explicitly designed as a partially-observable tabular/state-vector task. Its purpose is to evaluate Recurrent Neural Networks (RNNs) and memory-augmented agents.

To achieve this, the environment uses a custom Gymnasium `ObservationWrapper` to completely replace the default visual output with a simplified, 1D float vector. 

All of this logic is encapsulated within:
> 📁 [`temporal_comp_gym/envs/hazardous_delivery/wrappers.py`](../temporal_comp_gym/envs/hazardous_delivery/wrappers.py)

---

## The Observation Wrapper

The `HazardousDeliveryObservationWrapper` intercepts the default visual observation from the core engine and reconstructs it into a flat `gym.spaces.Box` of floating-point numbers.

```python
class HazardousDeliveryObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, include_health_in_state=True, include_inventory_in_state=True):
        super().__init__(env)
        self.include_health = include_health_in_state
        self.include_inventory = include_inventory_in_state
        
        # Base: 2 (pos) + 4 (surroundings) + 1 (current square) = 7 dimensions
        obs_dim = 2 + 4 + 1
        
        # Conditionally expand the vector if memory helpers are enabled
        if self.include_health: obs_dim += 1
        if self.include_inventory: obs_dim += 1
            
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
```

Depending on how it is configured via its `__init__` parameters, the final observation vector presented to the agent will contain either 7, 8, or 9 values.

---

## Exact Structure of the Observation Space

The vector is structured into three distinct sections: Spatial Coordinates, Local Sensors, and Optional Memory States.

### 1. Spatial Coordinates (Indices 0 - 1)
Instead of a global map, the agent receives its absolute spatial coordinates. This provides global spatial awareness without revealing the locations of any hazards or objectives across the grid.

1.  **`X Coordinate`**: The agent's absolute X position.
2.  **`Y Coordinate`**: The agent's absolute Y position.

### 2. Local Proximity Sensors (Indices 2 - 6)
The agent is equipped with a "proximity scanner" that reads the identity of the tiles immediately adjacent to it, as well as the tile it is currently standing on. This allows the agent to navigate safely around immediate hazards.

3.  **`Sensor Up`**: Object at (x, y-1)
4.  **`Sensor Down`**: Object at (x, y+1)
5.  **`Sensor Left`**: Object at (x-1, y)
6.  **`Sensor Right`**: Object at (x+1, y)
7.  **`Current Square`**: Object at (x, y)

Here is how the wrapper extracts these specific values from the core `env.grid`:

```python
x, y = env.agent_pos

# Query the core grid directly
up = grid.get(x, y-1)
down = grid.get(x, y+1)
left = grid.get(x-1, y)
right = grid.get(x+1, y)
curr = grid.get(x, y)

surroundings = [
    get_type_val(up),
    get_type_val(down),
    get_type_val(left),
    get_type_val(right)
]
```

### Sensor Value Mapping
The wrapper maps the physical Minigrid objects into discrete floating-point integers using a `get_type_val` helper function inside the `observation` method.

*   `0.0`: Safe / Empty Floor / Wall
*   `1.0`: Damage Square
*   `2.0`: Death Square (Lava)
*   `3.0`: Generic Package or Destination (used when `task_dependencies=False`)
*   `4.0`: Level Goal

**Task Dependency Overrides:**
If the environment is configured with `task_dependencies=True`, the agent needs to know the exact identity of packages and destinations to sequence them properly. The wrapper dynamically shifts these mapped values to distinguish between specific sequence IDs:

```python
def get_type_val(obj):
    # ... basic object checks ...
    
    if obj.type == 'package':
        # E.g., 11.0 for Package 1, 12.0 for Package 2...
        if getattr(env, 'task_dependencies', False) and getattr(obj, 'seq_id', None) is not None:
            return 10.0 + obj.seq_id 
        return 3.0
        
    if obj.type == 'destination':
        # E.g., 21.0 for Destination 1, 22.0 for Destination 2...
        if getattr(env, 'task_dependencies', False) and getattr(obj, 'seq_id', None) is not None:
            return 20.0 + obj.seq_id
        return 3.0
    # ...
```

### 3. Optional Memory Toggles (Indices 7 - 8)
These final two values are conditionally appended to the end of the vector. Their purpose is to test the internal memory capacity of the agent.

8.  **`Health State`**: The agent's current health score (e.g., 10.0). 
    *   *If hidden (`include_health_in_state=False`):* The agent must actively remember how many damage squares it has stepped on to know if it is near death.
9.  **`Inventory State`**: The total number of packages the agent has successfully delivered.
    *   *If hidden (`include_inventory_in_state=False`):* The agent must explicitly memorize its delivery history to know when it is safe to enter the Level Goal without triggering the massive premature termination penalty.

```python
# Conditionally appending memory values inside the observation() method
if self.include_health:
    vec.append(float(env.health))
if self.include_inventory:
    # Uses delivered packages as a proxy for the inventory memory challenge
    vec.append(float(env.delivered_packages))
    
return np.array(vec, dtype=np.float32)
```
