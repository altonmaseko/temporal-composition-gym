# Deferred Maintenance Environment: Core Engine

The `DeferredMaintenanceEnv` defines the physics engine, state transitions, and complex reward logic for the Deferred Maintenance task. Unlike grid-based environments, it utilizes a continuous 2D space powered by `pymunk` for collision and movement, and `pygame` for rendering.

All the core logic detailed below is located in:
> 📁 [`temporal_comp_gym/envs/deferred_maintenance/core.py`](../temporal_comp_gym/envs/deferred_maintenance/core.py)

---

## 1. Initialization and Spaces (`__init__`)

When the environment is instantiated, it configures the physics parameters, temporal rules, and the Gymnasium observation and action spaces.

### Action Space
The agent controls the rocket using two continuous inputs, each bounded between -1.0 and 1.0. 
*   `Action[0]`: Thrust (positive) or Air Brakes (highly negative).
*   `Action[1]`: Torque (rotation left or right).

```python
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

### Observation Space
Because this is a continuous physics task, the observation space is a complex `Dict` space providing precise telemetry, radar vectors to objectives, and localized collision sensors (`micro_nav`).

```python
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

---

## 2. Stage Generation and Reset (`reset` & `generate_stage`)

At the start of an episode, `reset()` clears the agent's inventory, calculates temporal delays (like `readiness`), and triggers `generate_stage()`. 

The environment can load static, pre-defined levels via `CUSTOM_STAGES`, or dynamically build procedurally generated levels using `_generate_procedural_stage()`.

```python
def generate_stage(self):
    # Clear old physics bodies from previous stages
    for ent in self.entities:
        if ent.body in self.space.bodies:
            self.space.remove(ent.body, ent.shape)
    self.entities.clear()
    
    # Determine layout
    if self.stages_traversed < len(CUSTOM_STAGES):
        stage_layout = CUSTOM_STAGES[self.stages_traversed]
    else:
        stage_layout = self._generate_procedural_stage()
        
    # Iterate through layout and instantiate Pymunk objects
    for row_idx, row in enumerate(stage_layout):
        for col_idx, val in enumerate(row):
            # ... calculates x, y based on cell_size ...
            if val == 1:
                self.entities.append(PathBoundary(self.space, rect))
            elif isinstance(val, str) and val.startswith("station_"):
                type_id = int(val.split("_")[1])
                self.entities.append(MaintenanceStation(self.space, rect, color_code=type_id))
            # ... handles Code Zones, Exits, Entries ...
```

---

## 3. Processing Actions and State Transitions (`step`)

The `step` method processes continuous physics updates, robust collision detection, and complex temporal task logic.

### Physics and Movement
The environment converts the normalized actions into raw thrust and torque. It also includes an auto-stabilization mechanic to keep the rocket upright if torque is neutral.

```python
def step(self, action):
    thrust = np.clip(action[0], -1.0, 1.0)
    torque_input = np.clip(action[1], -1.0, 1.0)
    
    # 1. Apply Thrust (or brakes)
    if thrust > 0:
        self.rocket.apply_thrust(thrust * 2000.0)
        
    # 2. Apply Rotation or Auto-Stabilization
    current_angle = self.rocket.body.angle
    if abs(torque_input) > 0.05:
        self.rocket.apply_rotation(torque_input * self.tilt_speed)
    else:
        restoring_torque = -current_angle * (self.tilt_speed * 0.8)
        self.rocket.apply_rotation(restoring_torque)
        
    # 3. Step the physical simulation forward
    self.space.step(1/60.0)
```

### Collision Detection via Shape Queries
Because standard objects in this environment are "sensors", they do not naturally bounce the rocket. The engine explicitly queries the `pymunk` space to find overlapping shapes to trigger game logic.

```python
    touching_station_id = None
    
    # Continuous Shape Query for robust trigger overlaps
    for info in self.space.shape_query(self.rocket.shape):
        shape = info.shape
        ent = next((e for e in self.entities if hasattr(e, 'shape') and e.shape == shape), None)
        
        if ent:
            if isinstance(ent, MaintenanceStation):
                touching_station_id = ent.color_code
            elif isinstance(ent, CodeZone):
                # Collect code logic
                if len(self.held_codes) < self.max_active_requests:
                    self.held_codes.append(ent.color_code)
                    self.pending_code_removals.append(ent)
            elif isinstance(ent, StageExit):
                self.reached_exit = True
```

### Hover Timers and Maintenance Logic
To successfully perform maintenance, the agent cannot just fly past a station. It must hover over it continuously for a set duration (`required_hover_duration`). Furthermore, the agent must be temporally "ready" (`readiness == 1.0`) and possess the matching code.

```python
    if touching_station_id is not None:
        if self.hovering_station_id != touching_station_id:
            self.hovering_station_id = touching_station_id
            self.hover_timer = 0.0  # Reset timer if station changes
            
        self.hover_timer += 1/60.0  # Increment timer for this physics step
        
        # Check if hover requirement is met
        if self.hover_timer >= self.required_hover_duration:
            # Check if agent has the code and is temporally ready
            if self.readiness == 1.0 and self.hovering_station_id in self.held_codes:
                reward += self.reward_service_station
                self.held_codes.remove(self.hovering_station_id)
            else:
                # Apply penalties for premature servicing or wrong codes
                if self.readiness < 1.0:
                    reward += self.penalty_premature_service
                elif self.hovering_station_id not in self.held_codes:
                    reward += self.penalty_wrong_station
            self.hover_timer = 0.0
```

### Stage Exits and Termination
The environment evaluates success or failure dynamically when the agent enters a `StageExit`. Leaving uncollected codes behind yields a massive penalty and terminates the run, whereas clearing a stage progresses the game until final perfect completion is achieved.

```python
    if self.reached_exit:
        self.reached_exit = False
        
        if self.uncollected_codes > 0:
            reward += self.penalty_leftover_codes
            terminated = True
        else:
            self.stages_traversed += 1
            if self.stages_traversed >= self.max_stages:
                # Finished the final stage!
                reward += self.reward_perfect_completion
                terminated = True
            else:
                # Progress to the next stage
                if self.stages_traversed >= self.readiness_target:
                    self.readiness = 1.0 
                self.generate_stage()
```
