# Deferred Maintenance Environment: Observation Space and Wrappers

In the Deferred Maintenance environment, the core engine computes a complex array of physical telemetry, spatial vectors, and temporal states, outputting them as a `gym.spaces.Dict`. However, most standard Reinforcement Learning (RL) algorithms (such as PPO or SAC) expect a flattened 1D tensor array as input.

To bridge this gap, the environment uses the `DeferredMaintenanceObservationWrapper` to aggregate and flatten the raw dictionary into a single continuous `gym.spaces.Box` vector.

All of this logic is encapsulated within:
> 📁 [`temporal_comp_gym/envs/deferred_maintenance/wrappers.py`](../temporal_comp_gym/envs/deferred_maintenance/wrappers.py)

---

## The Observation Wrapper

The wrapper calculates the final size of the 1D vector dynamically upon initialization. The total size scales based on the environment's configuration (specifically, the micro-navigation sensor type and the maximum number of code types allowed).

```python
class DeferredMaintenanceObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Determine the size of the micro-navigation sensors
        spatial_sensor_type = self.unwrapped.spatial_sensor_type
        if spatial_sensor_type == 'local_grid':
            micro_nav_size = 9 # 3x3 local binary grid
        else:
            micro_nav_size = 8 # 8 simulated ray-cast lines
            
        # Determine the size of station radar (X and Y per color)
        self.num_colors = self.unwrapped.max_code_types
        radar_size = self.num_colors * 2
        
        # Kinematics(6) + MicroNav(9/8) + MacroNav(2) + FloorCode(1) + Readiness(1) + Radar
        obs_size = 6 + micro_nav_size + 2 + 1 + 1 + radar_size
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
```

During each environment step, the `observation` method intercepts the raw `Dict` from `core.py`, extracts the NumPy arrays, flattens them, and sequentially concatenates them into the final continuous state vector:

```python
    def observation(self, obs):
        kinematics = np.array([
            obs['pos_x'], obs['pos_y'],
            obs['vel_x'], obs['vel_y'],
            obs['angle'], obs['ang_vel']
        ], dtype=np.float32)
        
        micro_nav = np.array(obs['micro_nav'], dtype=np.float32).flatten()
        macro_nav = np.array(obs['macro_nav'], dtype=np.float32).flatten()
        floor_code = np.array([obs['floor_code']], dtype=np.float32)
        readiness = np.array([obs['readiness']], dtype=np.float32)
        station_radar = np.array(obs['station_radar'], dtype=np.float32).flatten()
        
        return np.concatenate([
            kinematics, micro_nav, macro_nav, 
            floor_code, readiness, station_radar
        ])
```

---

## Exact Structure of the Observation Space

When evaluating or training an agent in this environment, it is critical to understand exactly what each segment of this flattened vector represents. The data is appended in the following strict order:

### 1. Kinematics (6 values)
Provides the core physical telemetry of the Rocket agent moving in continuous space.
*   `[0]` **`pos_x`**: Global X position.
*   `[1]` **`pos_y`**: Global Y position.
*   `[2]` **`vel_x`**: Current X axis velocity.
*   `[3]` **`vel_y`**: Current Y axis velocity.
*   `[4]` **`angle`**: Current rotation angle (in radians).
*   `[5]` **`ang_vel`**: Current rotational angular velocity.

### 2. Micro-Navigation Sensors (8 or 9 values)
Provides localized spatial awareness to help the agent avoid hard boundaries and navigate narrow safe pathways. The exact shape depends on the `spatial_sensor_type` parameter:
*   **`raycasts` (8 values)**: Simulates 8 lidar-like rays emitted radially (every 45 degrees) outward up to 150 pixels. Each value represents the normalized distance (`0.0` to `1.0`) to the first collision. `1.0` indicates open space/no obstacle.
*   **`local_grid` (9 values)**: Samples a 3x3 physical grid centered directly on the rocket. Each cell returns the `collision_type` integer ID of the shape occupying that space (e.g., `2.0` for Path Boundary, `7.0` for Stage Walls).

### 3. Macro-Navigation Waypoint (2 values)
*   **Reserved space**: Currently acts as a structural placeholder initialized to `[0.0, 0.0]`. Designed for future global direction hints (e.g., pointing a vector arrow toward the `StageExit`).

### 4. Current Floor Code (1 value)
Indicates what type of interactable zone the rocket is currently touching, serving as overlap confirmation for task logic.
*   `0.0`: The agent is in open space or not touching an interactable zone.
*   `> 0.0`: Returns the explicit `color_code` / Type ID (e.g., `1.0`, `2.0`) of the `MaintenanceStation` or `CodeZone` immediately beneath the agent.

### 5. Action Readiness Indicator (1 value)
A crucial temporal signal that tracks the core "deferral" mechanic, which locks the agent out of servicing stations until enough time (or stages) have passed.
*   `0.0 (WAITING)`: Agent is mathematically locked out. Attempting to hover and service a station right now will yield a premature-service penalty.
*   `1.0 (READY)`: Agent has survived the deferral period and is explicitly cleared to hover over matching stations to resolve codes and claim rewards.

### 6. Station Radar (`max_code_types * 2` values)
An internal compass dynamically sized based on the maximum number of code types allowed in the game config. It provides a direct tracking vector to all possible Maintenance Stations.
*   For each code type ID `i`, it appends two values: `[delta_x, delta_y]`.
*   This represents the exact relative offset (`station_x - rocket_x`, `station_y - rocket_y`) to the nearest station of that type.
*   If a station of type `i` does not physically exist on the current map layout, the radar gracefully defaults to a null-indicator vector of `[-99.0, -99.0]`.
