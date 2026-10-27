# Sequential Colour Environment: Core Engine

The `SequentialColourEnv` implements a visual memory and navigation task. It relies purely on Pygame to drive a 3-phase loop: **Memorisation**, **Retention**, and **Navigation**. The agent sees a sequence of colours, holds them in memory during a blank retention period, and then moves around a room touching the walls whose colours match the sequence, in order.

The core engine is defined in:
> 📁 [`temporal_comp_gym/envs/sequential_colour/core.py`](../temporal_comp_gym/envs/sequential_colour/core.py)

---

## What Changed in Protocol v2

If you read an earlier version of this page, these are the behaviour changes:

*   **Explicit events.** Every `step` and `reset` now returns `info["events"]`: a `cue` event for every colour of the sequence (the first one in the reset `info`, and repeated colours each get their own), `delay_start`, `nav_start`, `wall_touch` with the touched colour, and `room_reset`. Before, `info` was always empty.
*   **Termination causes.** The last step carries `info["termination_cause"]` (`success`, `wrong_wall_death`, `timeout`) and `info["is_success"]`.
*   **Internal episode limit.** Episodes now truncate at `max_steps=300`. Before, the environment had no limit of its own.
*   **No hidden answer in `info`.** During navigation, `info` never contains the target colour or the sequence (only with `debug_info=True`, which training and evaluation never use).
*   **Cue duration is a constructor argument.** `frames_per_colour` (default 5, the same as the old hard-coded value).
*   **Constructor validation.** Invalid `sequence_length_range`, a vocabulary below 2 without consecutive duplicates, or an unknown `reward_density` now raise `ValueError`.
*   **Reward is always a Python `float`.**
*   **The noise wrapper has its own seeded generator** (see [`11_sequential_colour_wrappers.md`](11_sequential_colour_wrappers.md)).

Unchanged: the three phases and their timing, the room, movement, wall-colour rules and all reward values.

---

## 1. Initialization and Configuration (`__init__`)

When the environment is instantiated, it validates and stores the task settings and builds the room.

### Configuration Parameters

| Parameter | Default | Effect |
|---|---|---|
| `render_mode` | `None` | `"human"` opens a Pygame window. (`"rgb_array"` is listed in `metadata` but `render()` returns nothing for it.) |
| `sequence_length_range` | `(3, 5)` | Inclusive range of sequence lengths, drawn uniformly. |
| `retention_delay_steps` | `2` | Number of black frames between the last cue and navigation. `0` skips retention. |
| `allow_consecutive_duplicates` | `False` | If `True`, the same colour can appear twice in a row. |
| `colour_vocabulary_size` | `6` | Number of colours used, taken in order from a fixed list of 16 (capped at 16). At 6: red, green, blue, yellow, magenta, cyan. |
| `randomise_wall_colors` | `True` | If `True`, wall colours are redrawn for each target; if `False`, walls use the first four vocabulary colours (see Room Colours). |
| `hard_death_mode` | `False` | If `True`, a wrong wall ends the episode. |
| `infinite_mode` | `False` | If `True`, completing a sequence starts a new one instead of ending the episode. Not used or validated in protocol v2. |
| `agent_wall_distance_blocks` | `6` | Room scale: blocks between the agent's spawn and each wall. |
| `reward_density` | `'sparse'` | `'sparse'`: reward only on completion. `'dense'`: reward on each correct wall. |
| `reward_correct_wall` | `1.0` | Per correct wall, dense mode only. |
| `reward_sequence_completion` | `10.0` | On completing the sequence, sparse mode only. |
| `penalty_severity` | `1.0` | Subtracted from the reward for every wrong wall (both modes). |
| `frames_per_colour` | `5` | Observations each cue colour stays on screen. |
| `max_steps` | `300` | Episode limit (truncation). `None` disables it. |
| `debug_info` | `False` | Adds `info["debug"]` with the hidden sequence. Tests and oracles only. |

The protocol v2 main condition (passed explicitly by `agents/common/env_factory.py`) uses exactly these defaults.

```python
        super().__init__()
        if sequence_length_range[0] < 1 or sequence_length_range[1] < sequence_length_range[0]:
            raise ValueError("sequence_length_range must satisfy 1 <= min <= max")
        if not allow_consecutive_duplicates and colour_vocabulary_size < 2:
            raise ValueError("colour_vocabulary_size must be >= 2 without consecutive duplicates")
        if reward_density not in ("sparse", "dense"):
            raise ValueError("reward_density must be 'sparse' or 'dense'")
        self.frames_per_colour = int(frames_per_colour)
        self.max_steps = max_steps
        self.debug_info = debug_info
```

### Observation and Action Spaces
This is a **vision-based** environment. The observation is an 84×84 RGB image, designed for Convolutional Neural Networks (CNNs). The action space is discrete 4-way movement.

```python
        # Action space: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 84x84x3 RGB image
        self.obs_size = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8
        )
```

Actions only have an effect during navigation; during memorisation and retention they are ignored.

The engine also keeps a `score` (starting at 10 and reduced by `penalty_severity` for each wrong wall). It is shown only in the render panel; it is not part of the reward or observation.

---

## 2. Sequence Generation and Reset (`_draw_sequence` & `reset`)

When `reset()` is called, the environment draws a new sequence, puts the agent in the centre, draws a room and starts the memorisation phase.

```python
    def _draw_sequence(self):
        seq_len = self.np_random.integers(self.sequence_length_range[0], self.sequence_length_range[1] + 1)
        self.sequence = []
        last_color_idx = -1
        for _ in range(seq_len):
            if self.allow_consecutive_duplicates:
                idx = int(self.np_random.integers(0, self.colour_vocabulary_size))
            else:
                choices = [i for i in range(self.colour_vocabulary_size) if i != last_color_idx]
                idx = int(self.np_random.choice(choices))
            self.sequence.append(self.vocab_colors[idx])
            last_color_idx = idx
        self.current_seq_idx = 0
        self.retention_counter = 0
        self.memorisation_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 10.0
        self.current_phase = "memorisation"
        self.step_count = 0
        self._draw_sequence()

        self.agent.reset()
        self._randomise_room()

        if self.render_mode == "human":
            self.render()

        # The first cue is on screen in the reset observation, so its event is in reset info.
        info = {"events": [("cue", self.colour_id(self.sequence[0]))]}
        if self.debug_info:
            info["debug"] = self._debug()
        return self._get_obs(), info
```

Colours that are not consecutive may repeat (for example red, blue, red).

### Seeding
The sequence and all room colours come from the environment's own generator, `self.np_random`. `reset(seed=s)` gives the same sequence, rooms and (for the same actions) the same frames every time; `reset()` without a seed continues the stream.

### Colour IDs
Events and debug information refer to colours by their index in the vocabulary:

```python
    def colour_id(self, rgb):
        """Index of an RGB colour in the vocabulary (the symbolic label used in events)."""
        return self.vocab_colors.index(tuple(rgb))
```

---

## 3. Room Colours (`_randomise_room`)

A room is drawn at reset, when navigation starts, and after every correct, non-final wall. Each time, the current target colour is guaranteed to be on at least one wall.

```python
    def _randomise_room(self):
        if self.randomise_wall_colors:
            if self.colour_vocabulary_size >= 4:
                wall_colors = list(self.np_random.choice(range(self.colour_vocabulary_size), size=4, replace=False))
                wall_colors = [self.vocab_colors[i] for i in wall_colors]
            else:
                wall_colors = [self.vocab_colors[self.np_random.integers(0, self.colour_vocabulary_size)] for _ in range(4)]
                
            target_color = self.sequence[self.current_seq_idx]
            if target_color not in wall_colors:
                wall_colors[self.np_random.integers(0, 4)] = target_color
        else:
            wall_colors = list(self.vocab_colors[:4])
            target_color = self.sequence[self.current_seq_idx]
            if target_color not in wall_colors:
                wall_colors[self.np_random.integers(0, 4)] = target_color
                
        self.wall_colors = {
            "top": wall_colors[0],
            "bottom": wall_colors[1],
            "left": wall_colors[2],
            "right": wall_colors[3]
        }
```

*   **Main condition (random colours, vocabulary ≥ 4):** the four walls get four different colours, exactly one of which is the target, at a random position.
*   **Random colours, vocabulary below 4:** colours are drawn with repetition, so the target can be on more than one wall.
*   **Fixed colours (`randomise_wall_colors=False`):** walls show the first four vocabulary colours in the order top, bottom, left, right; if the target is not among them, it replaces a random wall. This setting needs a vocabulary of at least 4: with fewer colours `reset()` fails with an `IndexError`.

---

## 4. The Three Phases (`step` and `_get_obs`)

The `_get_obs` method changes the image depending on the phase, and `step()` advances the phase. Events are emitted on the step whose resulting observation first shows them.

### Phase 1: Memorisation
The whole screen is filled with the current cue colour. Each colour is shown for `frames_per_colour` observations, starting with the reset observation.

```python
        if self.current_phase == "memorisation":
            color = self.sequence[self.current_seq_idx]
            obs = np.full((self.obs_size, self.obs_size, 3), color, dtype=np.uint8)
```

```python
        if self.current_phase == "memorisation":
            self.memorisation_timer += 1
            if self.memorisation_timer >= self.frames_per_colour:
                self.memorisation_timer = 0
                self.current_seq_idx += 1
                if self.current_seq_idx >= len(self.sequence):
                    self.current_seq_idx = 0
                    if self.retention_delay_steps > 0:
                        self.current_phase = "retention"
                        events.append(("delay_start", None))
                    else:
                        self.current_phase = "navigation"
                        self.agent.reset()
                        self._randomise_room()
                        events.append(("nav_start", None))
                else:
                    events.append(("cue", self.colour_id(self.sequence[self.current_seq_idx])))
```

### Phase 2: Retention
The screen is black for `retention_delay_steps` observations.

```python
        elif self.current_phase == "retention":
            obs = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
```

```python
        elif self.current_phase == "retention":
            self.retention_counter += 1
            if self.retention_counter >= self.retention_delay_steps:
                self.current_phase = "navigation"
                self.agent.reset()
                self._randomise_room()
                events.append(("nav_start", None))
```

### Timing at the Default Settings
For a sequence of length `L` (with `frames_per_colour = 5` and `retention_delay_steps = 2`):

| Step | Observation after the step | Event |
|---|---|---|
| reset | cue 1 | `cue(c1)` in reset `info` |
| 5, 10, …, 5(L−1) | next cue | `cue(c_k)` |
| 5L | black | `delay_start` |
| 5L + 1 | black | – |
| 5L + 2 | room | `nav_start` |
| from 5L + 3 | room; actions now move the agent | `wall_touch`, `room_reset` |

### Phase 3: Navigation
The observation is a top-down view of the room scaled to 84×84: black background, four coloured walls and the agent as a white square.

```python
        else:
            surf = pygame.Surface((self.obs_size, self.obs_size))
            surf.fill((0, 0, 0))
            
            scale_x = self.obs_size / self.game_size
            scale_y = self.obs_size / self.game_size
            
            rects = self.room.get_wall_rects()
            for key, rect in rects.items():
                scaled_rect = pygame.Rect(rect.x * scale_x, rect.y * scale_y, rect.width * scale_x, rect.height * scale_y)
                pygame.draw.rect(surf, self.wall_colors[key], scaled_rect)
                
            agent_rect = self.agent.get_rect()
            scaled_agent = pygame.Rect(agent_rect.x * scale_x, agent_rect.y * scale_y, agent_rect.width * scale_x, agent_rect.height * scale_y)
            pygame.draw.rect(surf, (255, 255, 255), scaled_agent)
            
            obs = pygame.surfarray.array3d(surf)
            obs = np.transpose(obs, (1, 0, 2)) # Shape is (84, 84, 3)
```

Each action moves the agent, then the engine checks for a wall overlap:

```python
        elif self.current_phase == "navigation":
            self.agent.move(action)
            
            agent_rect = self.agent.get_rect()
            wall_rects = self.room.get_wall_rects()
            
            hit_wall = None
            for key, rect in wall_rects.items():
                if agent_rect.colliderect(rect):
                    hit_wall = key
                    break
```

At the default room scale, the agent touches a wall on the 13th step of moving straight towards it. A perfect episode therefore takes `5L + 2 + 13L` steps: 56, 74 or 92 steps for sequence lengths 3, 4 and 5, well inside the 300-step limit.

---

## 5. Wall Touches and Rewards

When the agent touches a wall, the engine records a `wall_touch` event with the touched wall's colour (which is visible in the frame) and compares it with the current target. The target itself is never reported.

```python
            if hit_wall:
                touched_color = self.wall_colors[hit_wall]
                target_color = self.sequence[self.current_seq_idx]
                # The touched wall's colour is visible in the frame; the target is not reported.
                events.append(("wall_touch", self.colour_id(touched_color)))

                if touched_color == target_color:
                    if self.reward_density == 'dense':
                        reward += self.reward_correct_wall
                    self.current_seq_idx += 1

                    if self.current_seq_idx >= len(self.sequence):
                        if self.reward_density == 'sparse':
                            reward += self.reward_sequence_completion
                        if self.infinite_mode:
                            # Start next loop immediately (not used by the v2 baseline)
                            self.current_phase = "memorisation"
                            self._draw_sequence()
                            events.append(("cue", self.colour_id(self.sequence[0])))
                        else:
                            terminated = True
                            cause = "success"
                    else:
                        self.agent.reset()
                        self._randomise_room()
                        events.append(("room_reset", None))
                else:
                    self.score -= self.penalty_severity
                    reward -= self.penalty_severity
                    if self.hard_death_mode:
                        terminated = True
                        cause = "wrong_wall_death"
                    else:
                        self.agent.reset()
```

| Wall touched | Sparse reward (default) | Dense reward | What happens next | Events |
|---|---|---|---|---|
| Correct, not the last colour | `0` | `+1` | Agent re-centred, new room for the next target | `wall_touch(c)`, `room_reset` |
| Correct, last colour | `+10` | `+1` | Episode ends, `success` | `wall_touch(c)` |
| Wrong | `-1` | `-1` | Agent re-centred, **same room**, same target | `wall_touch(c)` |
| Wrong, `hard_death_mode=True` | `-1` | `-1` | Episode ends, `wrong_wall_death` | `wall_touch(c)` |

In dense mode there is no extra completion bonus. With `infinite_mode=True`, completing a sequence gives the same reward but starts a new memorisation phase (with a `cue` event) instead of ending the episode; the agent is not re-centred until the next navigation phase.

### Episode Limit and `info`

```python
        if (not terminated and self.max_steps is not None
                and self.step_count >= self.max_steps):
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

The limit lives inside the environment; the registered environment has no `max_episode_steps`, so `gym.make` does not add a `TimeLimit`. The parameter is called `max_steps` on purpose: `gym.make` would consume a `max_episode_steps` argument and add its own `TimeLimit`, bypassing the environment's `timeout` cause.

With `debug_info=True`, every `info` also carries the hidden state (used only by tests and the scripted oracle):

```python
    def _debug(self):
        return {
            "phase": self.current_phase,
            "sequence_ids": [self.colour_id(c) for c in self.sequence],
            "progress": int(self.current_seq_idx),
            "target_id": self.colour_id(self.sequence[self.current_seq_idx])
            if self.current_seq_idx < len(self.sequence) else None,
            "wall_ids": {k: self.colour_id(v) for k, v in self.wall_colors.items()},
        }
```

---

## 6. Rendering

With `render_mode="human"`, a 1200×800 Pygame window shows the full-size room (or the cue colour / black screen) and a side panel with the phase, score and progress. The panel shows hidden task state and is for humans only. `ICON_URLS` at the top of `core.py` can point the legend icons at image URLs; when empty (the default) `pytablericons` icons are used.
