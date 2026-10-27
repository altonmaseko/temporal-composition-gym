# Hazardous Delivery Environment: Core Engine

The `HazardousDeliveryCoreEnv` defines the fundamental physics, rules, state transitions, and reward logic for the Hazardous Delivery task. Built on top of `MiniGridEnv`, it implements a gridworld logistics challenge where an agent must navigate hazards, manage its health, pick up packages one at a time, drop each one at a destination, and only then enter the goal.

All the core logic detailed below is located in:
> 📁 [`temporal_comp_gym/envs/hazardous_delivery/core.py`](../temporal_comp_gym/envs/hazardous_delivery/core.py)

---

## What Changed in Protocol v2

If you read an earlier version of this page, these are the behaviour changes:

*   **Ordered deliveries removed.** `task_dependencies=True` now raises `NotImplementedError`. Packages and destinations are never numbered, and any package can go to any destination.
*   **Explicit events and outcomes.** Every `step` and `reset` returns `info["events"]`, and the last step of an episode carries `info["termination_cause"]` and `info["is_success"]`. Before, `info` was always empty, and lava deaths could only be told apart from premature goal entries by guessing from the reward.
*   **Truncation is never termination.** Reaching `max_steps` (500) sets `truncated=True` only if the episode did not also terminate on that step. Before, both flags could be `True` at once. The trainers no longer add their own `TimeLimit(1000)` on top.
*   **Unsolvable layouts are redrawn.** A layout in which lava walls off any package, destination or the goal is regenerated (from the same seeded random stream). Before, such layouts were kept and could not be completed. At the default settings this is rare: 5 of the seeds 0–999 need one redraw.
*   **Fixed start cell is reserved.** With `randomise_start=False`, the start cell `(1, 1)` is now claimed before objects are placed, so no object can be placed underneath the agent.
*   **Carrying capacity is validated.** `carrying_capacity < 1` raises `ValueError`.
*   **Reward is always a Python `float`.**
*   **Baseline observation hides the delivered count.** This is a wrapper setting, not an engine change; see [`05_hazardous_delivery_wrappers.md`](05_hazardous_delivery_wrappers.md).

Unchanged on purpose: packages and destinations share one sensor value, walls read as empty floor, and a cell's effect applies again when the agent bumps into a wall and stays on that cell.

---

## 1. Initialization and Configuration (`__init__`)

When the environment is instantiated, the `__init__` method validates the configuration and sets up the constraints, rewards, and internal state trackers for the agent.

### Configuration Parameters

| Parameter | Default | Effect |
|---|---|---|
| `grid_size` | `10` | Width and height of the grid, including the outer wall (an 8×8 interior). |
| `percent_death_squares` | `0.05` | Lava count = `int(interior cells × fraction)` (3 at the default size). |
| `percent_damage_squares` | `0.05` | Damage-square count, computed the same way (3 at the default size). |
| `randomise_start` | `True` | `True`: agent starts on a random free cell. `False`: agent starts at `(1, 1)`. |
| `randomise_goal` | `True` | `True`: goal on a random free cell. `False`: goal at `(grid_size−2, grid_size−2)`. |
| `num_packages` | `3` | Number of packages, and of destinations. All must be delivered before the goal. |
| `carrying_capacity` | `1` | Most packages carried at once. Must be ≥ 1. |
| `health_regeneration_nodes` | `2` | Number of single-use health nodes. |
| `task_dependencies` | `False` | Must stay `False`; `True` raises `NotImplementedError`. |
| `reward_delivery` | `10.0` | Reward for each delivery. |
| `reward_task_completion` | `50.0` | Reward for entering the goal after all deliveries. |
| `penalty_damage` | `-1.0` | Reward for each step that ends on a damage square. |
| `penalty_premature_goal` | `-50.0` | Reward for entering the goal before all deliveries. |
| `penalty_death_square` | `-50.0` | Reward for entering lava; also added when damage reduces health to 0. |
| `max_steps` | `500` | Episode limit; reaching it truncates the episode. |
| `resample_unsolvable` | `True` | Redraw layouts where lava walls off an objective. |
| `max_layout_attempts` | `200` | Layout draws allowed per reset before `RuntimeError`. |
| `debug_info` | `False` | Adds `info["debug"]` with hidden state. For tests and oracles only. |
| `**kwargs` | – | Passed to `MiniGridEnv` (for example `render_mode`). These only affect rendering. |

Maximum health is fixed at 10 in the code and is not a parameter.

The baseline experiments do not rely on these defaults implicitly: `agents/common/env_factory.py` passes the frozen main-condition settings explicitly. For the core engine they are the same values as the defaults above.

### Validation and State Variables

```python
        if task_dependencies:
            # Ordered deliveries are not part of the v2 baseline; the reward machine and the
            # counterfactual rules have not been validated for them.
            raise NotImplementedError("task_dependencies=True is not supported in protocol v2")
        if carrying_capacity < 1:
            raise ValueError("carrying_capacity must be >= 1")
        self.resample_unsolvable = resample_unsolvable
        self.max_layout_attempts = max_layout_attempts
        self.debug_info = debug_info
        self.layout_attempts = 0
        # ... parameter assignments ...

        self.max_health = 10
        self.health = self.max_health
        self.carried_packages = 0
        self.delivered_packages = 0
        
        mission_space = MissionSpace(mission_func=lambda: "Deliver all packages and reach the goal.")
        
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            **kwargs
        )
```

### Action Space Override
Unlike default Minigrid environments where actions include turning left/right and moving forward relative to the agent's orientation, this environment overrides the action space to strict 2D directional movement.

```python
        # Override action space to discrete directional: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = gym.spaces.Discrete(4)
```

`UP` decreases `y` and `DOWN` increases it. The agent's facing direction plays no part in movement or observation.

---

## 2. Grid Generation and Reset Logic (`reset` & `_gen_grid`)

### The `reset` Method
The `reset` method is called at the start of every new episode. It resets health, carried and delivered counts, lets the parent class generate the grid (through `_gen_grid`), and builds the reset `info`.

```python
    def reset(self, *, seed=None, options=None):
        self.health = self.max_health
        self.carried_packages = 0
        self.delivered_packages = 0
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["events"] = []
        info["layout_attempts"] = self.layout_attempts
        if self.debug_info:
            info["debug"] = self._task_state()
        return obs, info
```

The reset `info` always contains an empty `events` list and `layout_attempts` (how many layouts were drawn before an accepted one; usually 1).

### Seeding
All layout randomness comes from the environment's own generator, `self.np_random`. Calling `reset(seed=s)` reseeds it, so the same seed always gives the same layout and, for the same actions, the same rollout. Calling `reset()` without a seed continues the same stream. Redrawn layouts also come from this stream, so resampling stays reproducible.

### Layout Resampling (`_gen_grid`)
`_gen_grid` draws a layout and, when `resample_unsolvable=True`, keeps drawing until it is solvable:

```python
    def _gen_grid(self, width, height):
        # Layouts are drawn from self.np_random, so a seeded reset is reproducible. When
        # resampling is enabled, a layout is redrawn (from the same stream) until every
        # package, destination and the goal is reachable without crossing lava.
        for attempt in range(1, self.max_layout_attempts + 1):
            self.agent_pos = (-1, -1)
            self.agent_dir = -1
            self._gen_grid_once(width, height)
            self.layout_attempts = attempt
            if not self.resample_unsolvable or self.layout_is_solvable():
                return
        raise RuntimeError(f"no solvable layout after {self.max_layout_attempts} attempts")
```

A layout is solvable when a breadth-first search from the start reaches every package, destination and the goal without stepping on lava or walls. Damage squares count as passable (they cost health, not access). The goal ends the episode, so a path may end on it but not pass through it:

```python
    def layout_is_solvable(self):
        """True if all objectives are reachable from the start without entering lava.

        Damage squares are passable (they cost health, not access). The goal is terminal,
        so paths may end on it but not pass through it.
        """
        start = tuple(int(v) for v in self.agent_pos)
        targets = []
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type in ("package", "destination", "goal"):
                    targets.append((i, j))
        seen = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt in seen:
                    continue
                cell = self.grid.get(*nxt)
                if cell is not None and cell.type in ("wall", "lava"):
                    continue
                seen.add(nxt)
                if cell is None or cell.type != "goal":
                    queue.append(nxt)
        return all(t in seen for t in targets)
```

This check does not consider health. A layout counts as solvable even if every route would pass over more damage squares than the agent has health for; with 3 damage squares and 10 health at the default settings that cannot happen.

### Placing Objects (`_gen_grid_once`)
One layout draw places the outer wall, then the objects in a fixed order: the reserved start cell (fixed start only), the goal, packages, destinations, lava, damage squares, health nodes, and finally the agent (random start only). Each object goes on a random free cell.

```python
    def _gen_grid_once(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        num_inner = (width - 2) * (height - 2)
        num_death = int(num_inner * self.percent_death_squares)
        num_damage = int(num_inner * self.percent_damage_squares)

        if not self.randomise_start:
            # Reserve the fixed start cell before placing objects (place_obj skips agent_pos).
            self.agent_pos = (1, 1)
            self.agent_dir = 0

        if self.randomise_goal:
            self.place_obj(LevelGoal())
        else:
            self.put_obj(LevelGoal(), width - 2, height - 2)
            
        for i in range(self.num_packages):
            seq_id = (i + 1) if self.task_dependencies else None
            self.place_obj(Package(seq_id=seq_id))
            
        for i in range(self.num_packages):
            seq_id = (i + 1) if self.task_dependencies else None
            self.place_obj(Destination(seq_id=seq_id))
            
        for _ in range(num_death):
            self.place_obj(DeathSquare())
            
        for _ in range(num_damage):
            self.place_obj(DamageSquare())
            
        for _ in range(self.health_regeneration_nodes):
            self.place_obj(HealthNode())
            
        if self.randomise_start:
            self.place_agent()
```

Because `task_dependencies` is always `False` here, `seq_id` is always `None`.

---

## 3. Processing Actions and State Transitions (`step`)

The `step` method is the heart of the engine. It takes an action, attempts to move the agent, applies the effect of the cell the agent ends up on, records events, computes the reward, checks the episode limit, and returns the observation.

### Movement Logic
The action translates to an X,Y coordinate change. The agent moves only if the target cell is empty or its object allows overlapping. Walls are the only blocking objects; lava, damage, health, packages, destinations and the goal can all be entered.

```python
    def step(self, action):
        self.step_count += 1

        reward = 0.0
        terminated = False
        truncated = False
        events = []
        cause = None

        x, y = self.agent_pos
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
            
        fwd_cell = self.grid.get(x, y)
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = (x, y)
```

### Interaction, Rewards, and Events
After movement, the engine applies the effect of the cell the agent now occupies. If the move was blocked by a wall, the agent stays where it is and **that cell's effect applies again** (for example, bumping into a wall while standing on a damage square costs another health point).

Each effect appends one or more `(name, arg)` events to `info["events"]`. The `arg` is always `None` in this environment. Events come from the transition itself, never from the reward value.

**Hazards and Health:**

```python
        curr_cell = self.grid.get(self.agent_pos[0], self.agent_pos[1])
        
        # Cell effects apply to the cell the agent occupies after the move. As before, an agent
        # that bumps into a wall stays on its cell and the cell's effect applies again.
        if curr_cell is not None:
            if curr_cell.type == 'lava':
                events.append((EV_LAVA, None))
                reward = self.penalty_death_square
                terminated = True
                cause = "lava"

            elif curr_cell.type == 'damage':
                events.append((EV_DAMAGE, None))
                reward = self.penalty_damage
                self.health -= 1
                if self.health <= 0:
                    events.append((EV_HEALTH_DEPLETED, None))
                    terminated = True
                    reward += self.penalty_death_square
                    cause = "damage_death"

            elif curr_cell.type == 'health':
                events.append((EV_HEAL, None))
                self.health = self.max_health
                self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
```

**Logistics (Packages and Destinations):**
The agent picks up a package automatically on entering it, if it has spare capacity. It delivers automatically on entering a destination, if it is carrying something. A contact event is recorded either way, so an agent that walks onto a package while full (or a destination while empty-handed) gets a `package_contact` (or `destination_contact`) event with no change and no reward, and the object stays on the grid.

```python
            elif curr_cell.type == 'package':
                events.append((EV_PACKAGE_CONTACT, None))
                if self.carried_packages < self.carrying_capacity:
                    self.carried_packages += 1
                    self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
                    events.append((EV_PICKUP, None))

            elif curr_cell.type == 'destination':
                events.append((EV_DESTINATION_CONTACT, None))
                if self.carried_packages > 0:
                    self.carried_packages -= 1
                    self.delivered_packages += 1
                    reward = self.reward_delivery
                    self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
                    events.append((EV_DELIVERY, None))
```

**Goal Resolution:**
Entering the goal always ends the episode. It is a success only if every package has been delivered.

```python
            elif curr_cell.type == 'goal':
                events.append((EV_GOAL_CONTACT, None))
                terminated = True
                if self.delivered_packages >= self.num_packages:
                    reward = self.reward_task_completion
                    cause = "success"
                else:
                    reward = self.penalty_premature_goal
                    cause = "premature_goal"
```

### Episode Limit, Info and Return Values

```python
        # The limit truncates only episodes that did not terminate on this step.
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            cause = "timeout"

        obs = self.gen_obs()
        info = {"events": events}
        if terminated or truncated:
            info["termination_cause"] = cause
            info["is_success"] = cause == "success"
        if self.debug_info:
            info["debug"] = self._task_state()

        return obs, float(reward), terminated, truncated, info
```

The limit lives inside the environment. The registered environment has no `max_episode_steps`, so `gym.make` does not add a `TimeLimit` wrapper.

### Summary of Outcomes

| Step ends on | Events | Reward | Ends episode? | `termination_cause` |
|---|---|---|---|---|
| Empty floor | – | `0` | no | – |
| Damage square, health stays > 0 | `damage` | `-1` | no | – |
| Damage square, health reaches 0 | `damage`, `health_depleted` | `-51` | terminated | `damage_death` |
| Lava | `lava` | `-50` | terminated | `lava` |
| Health node | `heal` | `0` | no | – |
| Package, spare capacity | `package_contact`, `pickup` | `0` | no | – |
| Package, full | `package_contact` | `0` | no | – |
| Destination, carrying | `destination_contact`, `delivery` | `+10` | no | – |
| Destination, empty-handed | `destination_contact` | `0` | no | – |
| Goal, all delivered | `goal_contact` | `+50` | terminated | `success` |
| Goal, deliveries missing | `goal_contact` | `-50` | terminated | `premature_goal` |
| Any non-terminal step that reaches `max_steps` | (as above) | (as above) | truncated | `timeout` |

`info["is_success"]` is `True` only for `success`. A perfect episode at the defaults earns `3 × 10 + 50 = 80`, minus 1 for each damage step.

### Debug Information
With `debug_info=True`, every `info` (reset and step) also carries the hidden task state:

```python
    def _task_state(self):
        return {
            "health": int(self.health),
            "carried": int(self.carried_packages),
            "delivered": int(self.delivered_packages),
            "num_packages": int(self.num_packages),
        }
```

This is off for all training and evaluation and is used only by tests and the scripted oracle (`temporal_comp_gym/oracles.py`).

---

## 4. Rendering

`get_full_render` and `get_pov_render` draw the agent as a truck icon and append a legend panel showing health, carried and delivered counts and an icon key. This panel is for humans watching the episode; it is not part of the agent's observation.

The combination of rules above (health management, navigation around hazards, and remembering whether a package is carried and how many have been delivered) is what drives the temporal composition challenge in this environment.
