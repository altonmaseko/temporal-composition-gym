# Hazardous Delivery Environment: Agent Observation Space and Wrappers

By default, the `minigrid` framework provides agents with a 3D ego-centric visual matrix (an image of what the agent sees in front of it). However, the Hazardous Delivery environment is explicitly designed as a partially-observable tabular/state-vector task. Its purpose is to evaluate Recurrent Neural Networks (RNNs) and memory-augmented agents.

To achieve this, the environment uses a custom Gymnasium `ObservationWrapper` to completely replace the default visual output with a simplified, 1D float vector.

All of this logic is encapsulated within:
> 📁 [`temporal_comp_gym/envs/hazardous_delivery/wrappers.py`](../temporal_comp_gym/envs/hazardous_delivery/wrappers.py)

---

## What Changed in Protocol v2

*   **The baseline observation has 8 values, not 9.** The protocol v2 main condition (decision D1) hides the delivered count: `include_inventory_in_state=False`, health still visible. With the count visible, the only thing the agent had to remember was whether it was carrying a package. The visible-count setting is kept only as an optional ablation (`inventory_visible`).
*   **The wrapper's own defaults did not change.** Calling `gym.make("TemporalComp/HazardousDelivery-v0")` with no arguments still gives the 9-value vector. The 8-value main condition comes from `agents/common/env_factory.py`, which passes `include_inventory_in_state=False` explicitly.
*   **Numbered sensor values are gone in practice.** Because `task_dependencies=True` now raises in the core engine, packages and destinations always read `3.0`. The old `11.0, 12.0, …` / `21.0, 22.0, …` values can no longer occur.

---

## How the Environment Is Assembled

The registered environment `TemporalComp/HazardousDelivery-v0` points to a small factory function in [`__init__.py`](../temporal_comp_gym/envs/hazardous_delivery/__init__.py). It removes the two observation options from the keyword arguments, builds the core engine with the rest, and wraps it:

```python
def HazardousDeliveryEnv(**kwargs):
    include_health = kwargs.pop('include_health_in_state', True)
    include_inventory = kwargs.pop('include_inventory_in_state', True)
    
    env = HazardousDeliveryCoreEnv(**kwargs)
    
    env = HazardousDeliveryObservationWrapper(
        env,
        include_health_in_state=include_health,
        include_inventory_in_state=include_inventory
    )
    
    return env
```

`gym.make` then adds its standard `PassiveEnvChecker` and `OrderEnforcing` wrappers. It does not add a `TimeLimit`; the 500-step limit is inside the core engine.

---

## The Observation Wrapper

The `HazardousDeliveryObservationWrapper` ignores the default visual observation from the core engine and builds a flat `gym.spaces.Box` of `float32` values directly from the engine's state.

```python
class HazardousDeliveryObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, include_health_in_state=True, include_inventory_in_state=True):
        super().__init__(env)
        self.include_health = include_health_in_state
        self.include_inventory = include_inventory_in_state
        
        # 2 for pos, 4 for surroundings, 1 for current square
        obs_dim = 2 + 4 + 1
        if self.include_health:
            obs_dim += 1
        if self.include_inventory:
            obs_dim += 1
            
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
```

Depending on these two options, the vector has 7, 8 or 9 values:

| `include_health_in_state` | `include_inventory_in_state` | Size | Used for |
|---|---|---|---|
| `True` | `False` | 8 | **Protocol v2 main condition** (all baseline agents) |
| `True` | `True` | 9 | Plain `gym.make` default; `inventory_visible` ablation |
| `False` | `False` | 7 | `health_hidden` ablation |
| `False` | `True` | 8 | Not used in the protocol |

The wrapper changes only the observation. Rewards, termination, truncation and `info` (including `info["events"]` and `info["termination_cause"]`) pass through from the core engine unchanged.

---

## Exact Structure of the Observation Space

The vector is structured into three sections: Spatial Coordinates, Local Sensors, and Optional Memory States.

### 1. Spatial Coordinates (Indices 0 - 1)
Instead of a global map, the agent receives its absolute grid coordinates. This provides global spatial awareness without revealing the locations of any hazards or objectives across the grid. Since walls read the same as empty floor on the sensors (see below), the coordinates are also how the agent knows where the border is.

1.  **`X Coordinate`**: The agent's absolute X position (`1` to `grid_size−2`).
2.  **`Y Coordinate`**: The agent's absolute Y position (`1` to `grid_size−2`).

### 2. Local Proximity Sensors (Indices 2 - 6)
The agent is equipped with a "proximity scanner" that reads the identity of the tiles immediately adjacent to it, as well as the tile it is currently standing on. This allows the agent to navigate safely around immediate hazards.

3.  **`Sensor Up`**: Object at (x, y-1)
4.  **`Sensor Down`**: Object at (x, y+1)
5.  **`Sensor Left`**: Object at (x-1, y)
6.  **`Sensor Right`**: Object at (x+1, y)
7.  **`Current Square`**: Object at (x, y)

Here is how the wrapper builds these values from the core `env.grid`:

```python
    def observation(self, obs):
        env = self.env.unwrapped
        x, y = env.agent_pos
        # ... get_type_val helper (below) ...
        grid = env.grid
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
        
        curr_val = get_type_val(curr)
        vec = [float(x), float(y)] + surroundings + [float(curr_val)]
```

Objects that are used up (picked-up packages, completed destinations, consumed health nodes) are removed from the grid, so the "Current Square" reads `0.0` straight after a pickup, delivery or heal.

### Sensor Value Mapping
The wrapper maps the Minigrid objects into discrete values using a `get_type_val` helper inside the `observation` method:

*   `0.0`: Empty floor, wall, **or health node**
*   `1.0`: Damage Square
*   `2.0`: Death Square (Lava)
*   `3.0`: Package **or** Destination (deliberately indistinguishable)
*   `4.0`: Level Goal

Health nodes fall through to the default `0.0`, so the agent cannot see them; it only notices one through the `heal` effect (health jumping back to 10, if health is in the observation).

```python
        def get_type_val(obj):
            if obj is None: return 0.0
            if obj.type == 'damage': return 1.0
            if obj.type == 'lava': return 2.0
            
            if obj.type == 'package':
                if getattr(env, 'task_dependencies', False) and getattr(obj, 'seq_id', None) is not None:
                    return 10.0 + obj.seq_id # E.g., 11.0 for Package 1
                return 3.0
                
            if obj.type == 'destination':
                if getattr(env, 'task_dependencies', False) and getattr(obj, 'seq_id', None) is not None:
                    return 20.0 + obj.seq_id # E.g., 21.0 for Destination 1
                return 3.0
                
            if obj.type == 'goal': return 4.0
            return 0.0 
```

The `task_dependencies` branches are still in the code but can no longer run, because the core engine refuses `task_dependencies=True`. Packages and destinations therefore always read `3.0`.

### 3. Optional Memory Toggles (Indices 7 - 8)
These final values are conditionally appended to the end of the vector. Their purpose is to test the internal memory capacity of the agent.

8.  **`Health State`**: The agent's current health (`10` at the start; `0` appears only in the final observation of a `damage_death`).
    *   *Shown in the main condition.*
    *   *If hidden (`include_health_in_state=False`):* The agent must remember how many damage squares it has stepped on to know if it is near death.
9.  **`Inventory State`**: The number of packages delivered so far (`0` to `num_packages`).
    *   *Hidden in the main condition.*
    *   *If hidden:* The agent must remember its delivery history to know when it is safe to enter the Level Goal without the `-50` premature-goal penalty.

```python
        if self.include_health:
            vec.append(float(env.health))
        if self.include_inventory:
            # Using delivered packages as proxy for inventory memory challenge
            vec.append(float(env.delivered_packages))
            
        return np.array(vec, dtype=np.float32)
```

The number of packages **currently carried** is never part of the observation, under any setting. In the main condition the agent must therefore remember both whether it is carrying a package and how many it has delivered. It must also remember where it has seen packages and destinations, since the sensors only cover adjacent cells.
