# Deferred Maintenance Environment: Core Engine

The `DeferredMaintenanceEnv` defines the physics, state transitions and reward logic for the Deferred Maintenance task. Unlike the grid-based environments, it uses a continuous 2D space powered by `pymunk` for collision and movement, and `pygame` for rendering.

The task in one sentence: fly a rocket along a winding path through several stages, collect the maintenance codes found in the early stages, and once the "readiness" signal switches on, hover over the matching station to service each code before leaving the last stage holding nothing.

All the core logic detailed below is located in:
> 📁 [`temporal_comp_gym/envs/deferred_maintenance/core.py`](../temporal_comp_gym/envs/deferred_maintenance/core.py)

---

## What Changed in Protocol v2

If you read an earlier version of this page, these are the behaviour changes:

*   **Seeded, solvable episode schedule.** At reset the environment draws a readiness stage `r` and one distinct code type for each stage before `r`. Codes appear only in those stages, and every stage contains exactly one station per station type. Before, each stage drew code types independently (duplicates were possible) and the number of stations per code type shrank with the stage index, so many episodes (about 58% in the pilot, per the previous agent's audit) could not be completed.
*   **Distractor stations implemented.** `distractor_stations=True` now places stations for every type, including types that were not issued. Before, the option was stored but had no effect. The protocol v2 main condition turns it on (decision D2).
*   **Hand-authored stages removed.** `CUSTOM_STAGES` is gone; every stage is generated.
*   **All randomness is seeded.** Layouts, the readiness stage and the code types come from the environment's own seeded generator (`self.np_random`), and a fresh physics space is created each episode. Before, generation used global NumPy randomness, so the same seed gave different layouts.
*   **`readiness_delay_range` is inclusive and validated.** The default `(1, 2)` gives the same readiness stages {1, 2} as the old exclusive `(1, 3)`. Ranges that could never switch readiness on before the final exit now raise `ValueError`.
*   **Macro waypoint implemented.** It points to the next path tile towards the exit. Before, it was always `[0, 0]`.
*   **Explicit events, causes and an internal step limit.** `info` now carries `events`, and the last step carries `termination_cause` and `is_success`. Episodes truncate at `max_steps=4000`. Before, `info` was empty and the environment never truncated.
*   **`score` counts each reward once.** It is now the plain running sum of step rewards (before, some rewards were added twice). `score` is only shown in the render panel.
*   **Sensors read the stage's cell map.** The off-path check and the local-grid sensor use a stored map of cell types; the radar reports the *nearest* station of each type (before: the first one found); sensor queries no longer hit the rocket itself.
*   **`duration_matching_mode=True` raises `NotImplementedError`.** It never had an effect.
*   **Interactables are spaced apart.** Code zones and stations are at least two cells apart from each other and from the exit.
*   **The baselines see a rescaled 28-value observation** (see [`08_deferred_maintenance_wrappers.md`](08_deferred_maintenance_wrappers.md)).

Unchanged: the physics (thrust, air brakes, auto-levelling, tilt limit, gravity), the reward values, the hover-to-service rule and the exit rules.

---

## 1. Initialization and Spaces (`__init__`)

When the environment is instantiated, it validates the configuration, stores the physics parameters, task rules and rewards, and defines the Gymnasium spaces.

### Configuration Parameters

| Parameter | Default | Main condition | Effect |
|---|---|---|---|
| `render_mode` | `None` | `None` | `"human"` opens a Pygame window. |
| `spatial_sensor_type` | `'local_grid'` | `'local_grid'` | Micro-navigation sensor: `'local_grid'` or `'raycasts'`. Anything else raises. |
| `gravity_scale` | `500.0` | `500.0` | Downward gravity (pixels/s²). |
| `path_complexity` | `0.5` | (default) | Minimum walk length per stage is `max(10, int(5 × path_complexity))`. Values below 2.2 (including the default) give 10, so it has no effect unless set to 2.2 or more. |
| `max_active_requests` | `3` | `3` | Validation only (see below). |
| `distractor_stations` | `False` | **`True`** | `True`: every stage has a station for every type. `False`: only for the issued types. |
| `readiness_delay_range` | `(1, 2)` | `(1, 2)` | Inclusive range for the readiness stage `r`. |
| `tilt_speed` | `50000.0` | (default) | Torque scale for rotation. |
| `enable_air_brakes` | `True` | (default) | Enables the air brake (see Physics). |
| `required_hover_duration` | `2.0` | `2.0` | Seconds of continuous overlap needed to service a station. |
| `min_stage_width`, `min_stage_height` | `10`, `10` | (default) | Minimum stage size in cells, before the boundary ring is added. |
| `max_stages` | `4` | `4` | Number of stages. Exiting the last one ends the episode. Must be ≥ 2. |
| `max_code_types` | `3` | `3` | Number of code/station types `K`. |
| `codes_per_collection_stage` | `1` | `1` | Codes placed in each stage before readiness. |
| `enforce_boundaries` | `True` | (default) | `True`: the boundary ring is solid. `False`: the rocket can pass through it. |
| `max_steps` | `4000` | `4000` | Episode limit (truncation). `None` disables it. |
| `debug_info` | `False` | `False` | Adds `info["debug"]` with hidden state. Tests and oracles only. |
| `reward_*`, `penalty_*` | see Rewards | (defaults) | Reward values. |

The "Main condition" column shows what `agents/common/env_factory.py` passes for the protocol v2 baselines; "(default)" means it does not pass that setting. Note the one difference from the code defaults: the main condition turns **`distractor_stations` on**.

Two parameters are accepted but cannot change behaviour under the v2 schedule:

*   `max_active_requests`: validation makes sure every issued code fits (`r_max × codes_per_collection_stage ≤ max_active_requests`), so the carrying limit never blocks a collection.
*   `allow_duplicate_codes` (default `False`): issued codes are always distinct types, so the rocket never meets a second code of a type it already holds.

`duration_matching_mode` must stay `False`; `True` raises `NotImplementedError`.

### Validation

```python
        if spatial_sensor_type not in ("local_grid", "raycasts"):
            raise ValueError("spatial_sensor_type must be 'local_grid' or 'raycasts'")
        if duration_matching_mode:
            raise NotImplementedError("duration_matching_mode is not implemented in protocol v2")
        if max_stages < 2:
            raise ValueError("max_stages must be >= 2")
        lo, hi = int(readiness_delay_range[0]), int(readiness_delay_range[1])
        if not (1 <= lo <= hi <= max_stages - 1):
            raise ValueError(
                f"readiness_delay_range={readiness_delay_range} must satisfy 1 <= min <= max <= max_stages-1 "
                f"({max_stages - 1}); otherwise readiness could never switch on before the final exit")
        if hi * codes_per_collection_stage > max_code_types:
            raise ValueError("readiness max * codes_per_collection_stage exceeds max_code_types (codes must be distinct)")
        if hi * codes_per_collection_stage > max_active_requests:
            raise ValueError("readiness max * codes_per_collection_stage exceeds max_active_requests (codes could not all be held)")
        if codes_per_collection_stage < 1:
            raise ValueError("codes_per_collection_stage must be >= 1")
```

### Action Space
The agent controls the rocket with two continuous inputs, each clipped to [-1.0, 1.0].
*   `Action[0]`: Thrust when positive; air brake when below -0.5; nothing in between.
*   `Action[1]`: Rotation torque (ignored when its magnitude is 0.05 or less, in which case the rocket levels itself).

```python
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

### Observation Space
The core engine produces a `Dict` observation. In practice it is always flattened by the wrapper described in [`08_deferred_maintenance_wrappers.md`](08_deferred_maintenance_wrappers.md).

```python
        micro_nav_size = 9 if self.spatial_sensor_type == 'local_grid' else 8
        self.observation_space = spaces.Dict({
            'pos_x': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'pos_y': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'vel_x': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'vel_y': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'angle': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'ang_vel': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'micro_nav': spaces.Box(0.0, float(max_code_types + 10), shape=(micro_nav_size,), dtype=np.float32),
            'macro_nav': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            'floor_code': spaces.Box(0.0, float(max_code_types), shape=(), dtype=np.float32),
            'readiness': spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            'station_radar': spaces.Box(-np.inf, np.inf, shape=(2 * max_code_types,), dtype=np.float32),
        })
```

Fixed constants: one cell is 50 pixels (`cell_size`), and one step is 1/60 s of simulated time (`dt`).

---

## 2. Episode Schedule and Reset (`reset`)

`reset()` reseeds the generator (if a seed is given), builds a new physics space, draws the episode schedule, clears all task state and generates the first stage.

```python
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        # A fresh physics space per episode, so no solver/contact state leaks across episodes.
        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity_scale)

        lo, hi = self.readiness_delay_range
        self.readiness_target = int(rng.integers(lo, hi + 1))
        n_issued = self.readiness_target * self.codes_per_collection_stage
        self.issued_codes = [int(t) + 1 for t in rng.choice(self.max_code_types, size=n_issued, replace=False)]
        cps = self.codes_per_collection_stage
        self.stage_code_types = {s: self.issued_codes[s * cps:(s + 1) * cps] for s in range(self.readiness_target)}
        self.station_types = list(self.global_code_types) if self.distractor_stations else sorted(self.issued_codes)
        # ... all task counters, timers and the rocket are cleared ...
        self.generate_stage()

        if self.render_mode == "human":
            self.render()

        info = {"events": []}
        if self.debug_info:
            info["debug"] = self._debug()
        return self._get_obs(), info
```

What the schedule means, at the default and main-condition settings (`max_stages=4`, `K=3`, one code per collection stage):

*   The readiness stage `r` is 1 or 2, equally likely. Stages are numbered from 0.
*   Stages `0 … r−1` are **collection stages**. Each holds one code of a distinct type, so the rocket collects `r` codes in total.
*   Readiness switches on as the rocket enters stage `r`. Stages `r … 3` are **service stages**; there are always at least two of them.
*   Every stage, collection or service, contains one station per station type (all three types in the main condition).

So a perfect run is always possible: collect each code, keep it until readiness is on, hover over its station in any service stage, and leave the last stage holding nothing.

### Seeding
All of the schedule and every stage layout come from `self.np_random`. `reset(seed=s)` gives the same schedule, layouts and (for the same actions) the same rollout every time; `reset()` without a seed continues the stream. The physics space is rebuilt on every reset, so no contact state carries over between episodes. The environment passes Gymnasium's `check_env`.

---

## 3. Stage Generation (`_generate_procedural_stage` & `generate_stage`)

Each stage is a self-avoiding random walk on the grid. Its first cell is the entry, its last cell the exit, and the code zones and stations sit on cells in between:

```python
    def _generate_procedural_stage(self, stage_index):
        """Return (stage_array, path_coords) for one stage of the episode schedule.

        Interactables (stations and codes) sit on interior path tiles, at least two cells
        (Chebyshev) from each other and from the exit, so hovering on one never touches
        another or the exit. The walk is redrawn (longer every 20 attempts) until they fit.
        """
        rng = self.np_random
        codes = list(self.stage_code_types.get(stage_index, []))
        items = [("station", t) for t in self.station_types] + [("code", t) for t in codes]
        base_length = max(10, int(5 * self.path_complexity))
        for attempt in range(400):
            coords = self._random_walk(base_length + 2 * (attempt // 20))
            if len(coords) < len(items) + 2:
                continue
            exit_c = coords[-1]
            interior = coords[1:-1]
            placed = []
            for idx in rng.permutation(len(interior)):
                if len(placed) == len(items):
                    break
                tile = interior[idx]
                if self._cheb(tile, exit_c) < 2:
                    continue
                if any(self._cheb(tile, p) < 2 for p, _ in placed):
                    continue
                placed.append((tile, items[len(placed)]))
            if len(placed) == len(items):
                break
        else:
            raise RuntimeError("could not generate a stage layout")
        # ... the walk is centred in a grid of at least min_stage_width x min_stage_height cells
        #     (with at least one empty cell of margin on each side) ...
```

The walk has 10 steps by default (11 cells) and can stop early if it boxes itself in; it is redrawn, getting longer by 2 every 20 attempts, until all items fit. Interactables are not kept away from the entry, so a station can be right next to the spawn cell.

`generate_stage()` then removes the previous stage's bodies, adds a ring of `StageBoundary` cells around the grid, creates one static body per non-empty cell, records each cell's type in `self.cell_types` (used by the off-path check and the local-grid sensor) and the path order in `self.path_cells` (used by the waypoint), and places the rocket at the entry's centre with zero velocity and angle. It also resets the hover timer. Positions in the observation are measured in this stage's own pixel coordinates, so they jump when a new stage starts.

---

## 4. Processing Actions and State Transitions (`step`)

The `step` method applies the action, advances the physics by one 1/60 s tick, then applies the task rules.

### Physics and Movement

```python
    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        thrust = float(np.clip(action[0], -1.0, 1.0))
        torque_input = float(np.clip(action[1], -1.0, 1.0))

        brake_engaged = (thrust < -0.5)

        if self.enable_air_brakes and brake_engaged and not self._prev_brake_engaged:
            self.rocket.body.velocity = (0.0, self.rocket.body.velocity.y)
            self.rocket.body.angular_velocity = 0.0

        self._prev_brake_engaged = brake_engaged

        if thrust > 0:
            self.rocket.apply_thrust(thrust * 2000.0)

        max_angle = math.pi / 2.0
        current_angle = self.rocket.body.angle

        if abs(torque_input) > 0.05:
            self.rocket.apply_rotation(torque_input * self.tilt_speed)
            self.rocket.body.angular_velocity *= 0.85
        else:
            restoring_torque = -current_angle * (self.tilt_speed * 0.8)
            self.rocket.apply_rotation(restoring_torque)
            self.rocket.body.angular_velocity *= 0.85

            if abs(current_angle) < 0.05 and abs(self.rocket.body.angular_velocity) < 0.1:
                self.rocket.body.angle = 0.0
                self.rocket.body.angular_velocity = 0.0

        if self.rocket.body.angle > max_angle:
            self.rocket.body.angle = max_angle
            self.rocket.body.angular_velocity = min(self.rocket.body.angular_velocity, 0.0)
        elif self.rocket.body.angle < -max_angle:
            self.rocket.body.angle = -max_angle
            self.rocket.body.angular_velocity = max(self.rocket.body.angular_velocity, 0.0)

        self.space.step(self.dt)
        self.step_count += 1
```

In words:
*   **Thrust**: a positive `Action[0]` pushes along the rocket's nose with force `2000 × thrust`. With mass 1 and gravity 500, a thrust of `0.25` exactly cancels gravity when upright.
*   **Air brake**: only on the step on which `Action[0]` first drops below -0.5 (not while it stays there), horizontal velocity and spin are set to zero. Vertical velocity is kept, so gravity still acts.
*   **Rotation**: a torque input above 0.05 in magnitude spins the rocket; otherwise a restoring torque levels it, and it snaps upright once nearly level. Spin is damped by 15% every step, and the tilt is limited to ±90°.

### Overlaps and Code Collection
After the physics tick, the engine finds every object the rocket's shape overlaps:

```python
        for q in self.space.shape_query(self.rocket.shape):
            ent = self.shape_to_entity.get(q.shape)
            if ent is None:
                continue
            if isinstance(ent, MaintenanceStation):
                touching_station_id = ent.color_code
            elif isinstance(ent, CodeZone):
                touching_code_id = ent.color_code
                if ent in self.pending_code_removals:
                    continue
                if len(self.held_codes) < self.max_active_requests and (
                        self.allow_duplicate_codes or ent.color_code not in self.held_codes):
                    self.held_codes.append(ent.color_code)
                    self.pending_code_removals.append(ent)
                    events.append(("code_collected", int(ent.color_code)))
            elif isinstance(ent, StageExit):
                reached_exit = True
            elif isinstance(ent, StageBoundary):
                touching_boundary = True

        self.touching_floor_code = float(touching_station_id or touching_code_id or 0.0)

        if touching_boundary:
            reward += self.penalty_touch_boundary
            events.append(("boundary_contact", None))
```

A code is collected on the first step the rocket touches it. The floor code reports the touched station's type, else the touched code's type, else 0.

### Hover Timers and Maintenance Logic
To service a station, the rocket must overlap the same station continuously for `required_hover_duration` (120 steps at the default 2.0 s). When the timer is reached, a `service_attempt` happens and the timer restarts, so staying longer produces another attempt every 120 steps.

```python
        if touching_station_id is not None:
            if self.hovering_station_id != touching_station_id:
                self.hovering_station_id = touching_station_id
                self.hover_timer = 0.0
            self.hover_timer += self.dt
            # small tolerance so 2.0 s means exactly 120 steps at 60 Hz
            if self.hover_timer >= self.required_hover_duration - 1e-9:
                k = int(self.hovering_station_id)
                events.append(("service_attempt", k))
                if self.readiness == 1.0 and k in self.held_codes:
                    reward += self.reward_service_station
                    self.held_codes.remove(k)
                    self.serviced_codes.append(k)
                elif self.readiness < 1.0:
                    reward += self.penalty_premature_service
                else:
                    reward += self.penalty_wrong_station
                self.hover_timer = 0.0
        else:
            self.hovering_station_id = None
            self.hover_timer = 0.0
```

### Off-Path Penalty and Tile Conversion
If the rocket's centre is over an empty (void) cell, the step costs `penalty_off_path`. Boundary cells count as on-path here, so touching a wall is not penalised twice. Collected code zones are then replaced by ordinary path tiles.

```python
        if self.cell_types.get(self._cell_of(self.rocket.body.position), CELL_VOID) not in SAFE_CELLS:
            reward += self.penalty_off_path
            events.append(("off_path", None))

        # Collected code zones turn into plain path tiles.
        for ent in self.pending_code_removals:
            if ent in self.entities:
                if ent.body in self.space.bodies:
                    self.space.remove(ent.body, ent.shape)
                self.entities.remove(ent)
                self.shape_to_entity.pop(ent.shape, None)
                tile = PathBoundary(self.space, ent.rect)
                self.entities.append(tile)
                self.shape_to_entity[tile.shape] = tile
                self.cell_types[(ent.rect[0] // self.cell_size, ent.rect[1] // self.cell_size)] = CELL_PATH
                self.uncollected_codes -= 1
        self.pending_code_removals.clear()
```

### Stage Exits and Termination
Touching the exit leaves the stage. Leaving a code uncollected in the current stage ends the episode as a failure. Leaving the last stage ends the episode, as a success only if no codes are still held.

```python
        if reached_exit:
            if self.uncollected_codes > 0:
                events.append(("stage_exit_incomplete", int(self.stages_traversed)))
                reward += self.penalty_leftover_codes
                terminated = True
                cause = "uncollected_codes_at_exit"
            else:
                events.append(("stage_exit", int(self.stages_traversed)))
                self.stages_traversed += 1
                if self.stages_traversed >= self.max_stages:
                    events.append(("final_exit", None))
                    if len(self.held_codes) > 0:
                        reward += self.penalty_leftover_codes
                        cause = "held_codes_at_final_exit"
                    else:
                        reward += self.reward_perfect_completion
                        cause = "success"
                    terminated = True
                else:
                    if self.stages_traversed >= self.readiness_target and self.readiness < 1.0:
                        self.readiness = 1.0
                        events.append(("readiness_on", None))
                    self.generate_stage()

        # score is the running sum of step rewards (each reward counted once)
        self.score += reward

        truncated = False
        if not terminated and self.max_steps is not None and self.step_count >= self.max_steps:
            truncated = True
            cause = "timeout"

        if self.render_mode == "human":
            self.render()

        info = {"events": events}
        if terminated or truncated:
            info["termination_cause"] = cause
            info["is_success"] = cause == "success"
        if self.debug_info:
            info["debug"] = self._debug()
        return self._get_obs(), float(reward), terminated, truncated, info
```

Because every code must be collected before its stage can be left, and success requires holding nothing at the final exit, success means every issued code was collected **and** serviced.

The step limit lives inside the environment. The registered environment has no `max_episode_steps`, so `gym.make` does not add a `TimeLimit`. The default of 4,000 steps (about 67 simulated seconds) is at least twice the 95th-percentile episode length of the scripted oracle (1,859 steps on the 200 final-evaluation seeds, as recorded in the protocol).

### Rewards

| Situation | Reward | Event | Ends episode? |
|---|---|---|---|
| Overlapping a stage boundary (per step) | `-5` | `boundary_contact` | no |
| Rocket centre over a void cell (per step) | `-0.1` | `off_path` | no |
| Touching a code zone | `0` | `code_collected(k)` | no |
| Hover timer reached, ready and code `k` held | `+100` | `service_attempt(k)` | no |
| Hover timer reached, not yet ready | `-10` | `service_attempt(k)` | no |
| Hover timer reached, ready but code `k` not held | `-10` | `service_attempt(k)` | no |
| Exit with a code left uncollected in this stage | `-1000` | `stage_exit_incomplete(s)` | terminated, `uncollected_codes_at_exit` |
| Exit of a non-final stage | `0` | `stage_exit(s)` (+ `readiness_on` when entering stage `r`) | no |
| Exit of the last stage, nothing held | `+1000` | `stage_exit(s)`, `final_exit` | terminated, `success` |
| Exit of the last stage, codes still held | `-1000` | `stage_exit(s)`, `final_exit` | terminated, `held_codes_at_final_exit` |
| Non-terminal step reaching `max_steps` | (as above) | (as above) | truncated, `timeout` |

Rewards on one step add up (for example, boundary contact while off-path). `s` is the 0-based index of the stage being left. A perfect episode earns `100 × r + 1000` (1,100 or 1,200 at the defaults) minus any per-step penalties.

### Events and `info`
Events within a step appear in this order: `code_collected`, `boundary_contact`, `service_attempt`, `off_path`, then `stage_exit_incomplete` or `stage_exit` followed by `final_exit` or `readiness_on`. Reset returns `{"events": []}`. With `debug_info=True`, every `info` also has `debug`:

```python
    def _debug(self):
        return {
            "held_codes": list(self.held_codes),
            "serviced_codes": list(self.serviced_codes),
            "issued_codes": list(self.issued_codes),
            "readiness_target": int(self.readiness_target),
            "stage": int(self.stages_traversed),
            "uncollected_codes": int(self.uncollected_codes),
        }
```

This is off for all training and evaluation; it is used by tests and the scripted oracle in `temporal_comp_gym/oracles.py`.

---

## 5. Building the Observation (`waypoint` & `_get_obs`)

The macro-navigation waypoint is the offset from the rocket to the centre of the next path cell towards the exit. Off the path, it is measured from the nearest path cell:

```python
    def waypoint(self):
        """Offset from the rocket to the centre of the next path tile towards the exit."""
        pos = np.array([self.rocket.body.position.x, self.rocket.body.position.y])
        idx = self.path_index.get(self._cell_of(pos))
        if idx is None:
            idx = int(np.argmin(((self._path_centers - pos) ** 2).sum(axis=1)))
        target = self._path_centers[min(idx + 1, len(self._path_centers) - 1)]
        return target - pos
```

`_get_obs` fills the dictionary: position and velocity (pixels, pixels/s), angle (radians) and angular velocity; the micro-navigation sensor; the waypoint; the floor code; readiness (`0.0` or `1.0`); and the station radar.

```python
        if self.spatial_sensor_type == 'local_grid':
            # Cell type of the 3x3 neighbourhood (same codes as the collision types).
            grid = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    cell = self._cell_of((pos.x + dx * self.cell_size, pos.y + dy * self.cell_size))
                    grid.append(float(self.cell_types.get(cell, CELL_VOID)))
            obs['micro_nav'] = grid
        else:
            ray_distances = []
            for i in range(8):
                angle = i * (2 * math.pi / 8)
                direction = pymunk.Vec2d(1, 0).rotated(angle)
                end = pos + direction * 150.0 # 150px sensor range
                res = self.space.segment_query_first(pos, end, 1, SENSOR_FILTER)
                ray_distances.append(float(res.alpha) if res else 1.0)
            obs['micro_nav'] = ray_distances

        obs['macro_nav'] = [float(v) for v in self.waypoint()]
        obs['floor_code'] = float(self.touching_floor_code)
        obs['readiness'] = float(self.readiness)

        radar = []
        for v in self.global_code_types:
            best = None
            for e in self.entities:
                if isinstance(e, MaintenanceStation) and e.color_code == v:
                    d = (e.body.position.x - pos.x, e.body.position.y - pos.y)
                    if best is None or d[0] ** 2 + d[1] ** 2 < best[0] ** 2 + best[1] ** 2:
                        best = d
            radar.extend(best if best is not None else [-99.0, -99.0])
        obs['station_radar'] = [float(v) for v in radar]
```

The exact layout of the flattened vector is described in [`08_deferred_maintenance_wrappers.md`](08_deferred_maintenance_wrappers.md).

---

## 6. Rendering

With `render_mode="human"`, a Pygame window shows the stage around the rocket and a side panel with the score, readiness, held codes, hover timer, stage count and uncollected codes. The panel shows hidden task state and is for humans only. `ICON_URLS` at the top of `core.py` can point the icons at image URLs; when empty (the default) `pytablericons` icons are used.
