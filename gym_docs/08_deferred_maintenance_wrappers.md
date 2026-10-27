# Deferred Maintenance Environment: Observation Space and Wrappers

In the Deferred Maintenance environment, the core engine computes physical telemetry, navigation hints and the readiness signal, and returns them as a `gym.spaces.Dict`. Most standard Reinforcement Learning (RL) algorithms (such as PPO) expect a flat 1D vector, so the environment flattens this dictionary with a wrapper. For the protocol v2 baselines, a second wrapper then rescales the vector to fixed, similar ranges.

All of this logic is encapsulated within:
> 📁 [`temporal_comp_gym/envs/deferred_maintenance/wrappers.py`](../temporal_comp_gym/envs/deferred_maintenance/wrappers.py)

---

## What Changed in Protocol v2

*   **The baselines see 28 values, not 25.** `agents/common/env_factory.py` adds a new wrapper, `MaintenanceScaledObservation`, on top of every maintenance environment. It applies fixed scaling (no running statistics to learn or save) and replaces the radar's `-99` "no station" marker with a presence flag. PPO, RecurrentPPO and PPO-RM all receive exactly the same vector.
*   **The macro waypoint is real.** Its two values used to be a constant `[0.0, 0.0]` placeholder; they now point to the next path cell towards the exit.
*   **The radar points to the nearest station of each type** (before: the first one found). In the main condition (distractor stations on) every type has a station in every stage, so the radar never reports "absent".
*   **The local-grid sensor reads the stage's cell map** rather than querying shapes, so it can no longer report the rocket itself, and empty cells always read `0`.
*   A plain `gym.make("TemporalComp/DeferredMaintenance-v0")` still returns the unscaled 25-value vector.

---

## How the Environment Is Assembled

The registered environment `TemporalComp/DeferredMaintenance-v0` points to a small class in [`__init__.py`](../temporal_comp_gym/envs/deferred_maintenance/__init__.py) that builds the core engine and wraps it with the flattening wrapper:

```python
class DeferredMaintenanceEnv(DeferredMaintenanceObservationWrapper):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, **kwargs):
        env = _CoreEnv(**kwargs)
        super().__init__(env)
```

`gym.make` adds its standard `PassiveEnvChecker` and `OrderEnforcing` wrappers (no `TimeLimit`; the step limit is inside the engine). The baseline factory then adds the scaling wrapper:

```python
    env = gym.make(ENV_IDS[env_key], **kwargs)
    if env_key == "maintenance":
        env = MaintenanceScaledObservation(env)
    return env
```

Neither wrapper changes rewards, termination, truncation or `info`.

---

## The Flattening Wrapper (`DeferredMaintenanceObservationWrapper`)

The wrapper works out the vector size when it is created. The size depends on the micro-navigation sensor type and the number of code types `K`:

```python
class DeferredMaintenanceObservationWrapper(gym.ObservationWrapper):
    """Flattens the core env's dict observation into the 1D vector described in
    agent_observation_space.txt (raw units: pixels, pixels/s, radians)."""

    def __init__(self, env):
        super().__init__(env)

        # Determine the size of the micro-navigation sensors
        spatial_sensor_type = self.unwrapped.spatial_sensor_type
        if spatial_sensor_type == 'local_grid':
            micro_nav_size = 9 # 3x3 local grid of cell types
        else:
            micro_nav_size = 8 # 8 simulated ray-cast lines

        # Determine the size of station radar (X and Y per color)
        self.num_colors = self.unwrapped.max_code_types
        radar_size = self.num_colors * 2

        # Kinematics(6) + MicroNav(9/8) + MacroNav(2) + FloorCode(1) + Readiness(1) + Radar
        obs_size = 6 + micro_nav_size + 2 + 1 + 1 + radar_size
        self.micro_nav_size = micro_nav_size

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
```

On every reset and step, the `observation` method concatenates the dictionary entries in a fixed order:

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

        flat_obs = np.concatenate([
            kinematics,
            micro_nav,
            macro_nav,
            floor_code,
            readiness,
            station_radar
        ])

        return flat_obs
```

With `local_grid` and `K = 3` this gives `6 + 9 + 2 + 1 + 1 + 6 = 25` values.

---

## Exact Structure of the Flattened (Unscaled) Vector

Indices below are for `local_grid` and `K = 3`.

### 1. Kinematics (indices 0–5)
*   `[0]` **`pos_x`**, `[1]` **`pos_y`**: Rocket position in the current stage's own coordinates (pixels; one cell = 50 px; y grows downwards). Each stage starts a new coordinate frame, so position jumps when a stage changes.
*   `[2]` **`vel_x`**, `[3]` **`vel_y`**: Velocity (pixels/s).
*   `[4]` **`angle`**: Tilt (radians, limited to ±π/2).
*   `[5]` **`ang_vel`**: Angular velocity (radians/s).

### 2. Micro-Navigation Sensors (indices 6–14 for `local_grid`)
The shape depends on `spatial_sensor_type`:
*   **`local_grid` (9 values, used by the baselines)**: The cell type of the 3×3 block of cells around the rocket's centre, row by row from top-left to bottom-right (the rocket's own cell is the middle value). Codes: `0` empty void (also anything outside the stage), `2` path (including collected code zones), `3` code zone, `4` station, `5` exit, `6` entry, `7` stage boundary.
*   **`raycasts` (8 values)**: 8 rays every 45° (starting along +x), each up to 150 px. Each value is the fraction of the ray length (`0.0` to `1.0`) at which it first hits any shape other than the rocket, or `1.0` if it hits nothing. All cells of the level, including path tiles, are shapes, so a ray starting inside a path tile reports `0.0`. In practice this sensor reads all zeros whenever the rocket's centre is on the path. It is not used by any baseline.

### 3. Macro-Navigation Waypoint (indices 15–16)
*   The offset `(dx, dy)` in pixels from the rocket to the centre of the next path cell towards the exit. If the rocket is off the path, it is measured from the path cell nearest the rocket. On the exit cell it points to the exit's centre.

### 4. Current Floor Code (index 17)
*   `0.0`: Not touching a station or a code zone.
*   `1.0 … K`: The type of the station (or, if none, the code zone) the rocket overlaps. A code zone shows its type only on the step it is collected; after that it is a path tile.

### 5. Action Readiness Indicator (index 18)
*   `0.0`: Before readiness. Completing a hover on a station now costs `-10` (premature service).
*   `1.0`: From the step on which the rocket enters the readiness stage `r` until the end of the episode. Hovering on the station of a held code now services it (`+100`).

### 6. Station Radar (indices 19–24, two values per type)
*   For each type `1 … K` in order: the offset `(station_x − rocket_x, station_y − rocket_y)` in pixels to the nearest station of that type in the current stage.
*   If the stage has no station of that type, the pair is `[-99.0, -99.0]`. This only happens with `distractor_stations=False`.

---

## The Scaling Wrapper (`MaintenanceScaledObservation`)

This wrapper is what every protocol v2 baseline sees. It rescales the flattened vector with fixed constants so all entries have similar ranges, and expands the radar to three values per type:

```python
class MaintenanceScaledObservation(gym.ObservationWrapper):
    """Fixed, deterministic rescaling of the flat maintenance observation (protocol v2).

    Applied identically for PPO, RecurrentPPO and PPO-RM, so no running statistics are
    learned or need saving. Positions and offsets are expressed in cells, velocities in
    cells/s / 4, angular velocity / 10, cell-type codes / 7, floor code / K. The radar's
    -99 "absent" sentinel is replaced by a presence flag plus a zero offset, giving three
    values per station type instead of two.
    """

    def __init__(self, env):
        super().__init__(env)
        core = env.unwrapped
        self.cell = float(core.cell_size)
        self.k = int(core.max_code_types)
        self.micro = 9 if core.spatial_sensor_type == "local_grid" else 8
        self.grid_sensor = core.spatial_sensor_type == "local_grid"
        size = 6 + self.micro + 2 + 1 + 1 + 3 * self.k
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)

    def observation(self, obs):
        c, k, m = self.cell, self.k, self.micro
        kin = obs[:6].astype(np.float64)
        kin = np.array([kin[0] / c, kin[1] / c, kin[2] / (4 * c), kin[3] / (4 * c), kin[4], kin[5] / 10.0])
        micro = obs[6:6 + m] / 7.0 if self.grid_sensor else obs[6:6 + m]
        macro = obs[6 + m:8 + m] / c
        floor = obs[8 + m:9 + m] / k
        ready = obs[9 + m:10 + m]
        radar_raw = obs[10 + m:10 + m + 2 * k].reshape(k, 2)
        radar = []
        for dx, dy in radar_raw:
            if dx == -99.0 and dy == -99.0:
                radar.extend([0.0, 0.0, 0.0])
            else:
                radar.extend([1.0, dx / c, dy / c])
        return np.concatenate([kin, micro, macro, floor, ready, radar]).astype(np.float32)
```

### Scaled Layout (main condition: `local_grid`, `K = 3`, 28 values)

| Indices | Content | Scaling |
|---|---|---|
| 0–1 | position | ÷ 50 (cells) |
| 2–3 | velocity | ÷ 200 (cells/s ÷ 4) |
| 4 | angle | unchanged (radians) |
| 5 | angular velocity | ÷ 10 |
| 6–14 | 3×3 cell types | ÷ 7 (so a boundary reads 1.0) |
| 15–16 | waypoint offset | ÷ 50 (cells) |
| 17 | floor code | ÷ K |
| 18 | readiness | unchanged (0 or 1) |
| 19–27 | radar, per type: `present, dx, dy` | present ∈ {0, 1}; offsets ÷ 50; absent → `0, 0, 0` |

With `raycasts`, the 8 sensor values are passed through unscaled and the vector has 27 values.

PPO-RM appends 8 reward-machine features to these 28 values; that is done by the agent's own wrapper, not by the environment.

---

## What the Agent Has to Remember

Nothing in the observation says which codes the rocket holds or which it has already serviced. In the main condition every stage shows a station of every type, so the stations on screen do not give this away either. The agent must remember which code types it collected in the early stages and carry that memory until readiness switches on. Readiness itself is visible.
