# Sequential Colour Environment: Core Engine

The `SequentialColourEnv` implements a strict visual memory and navigation task. Unlike the other environments, it relies purely on Pygame to drive a 3-phase temporal loop: **Memorisation**, **Retention**, and **Navigation**. The agent must visually memorize a sequence of colors, hold them in memory during a blank retention period, and then navigate a physical room to touch walls matching the sequence order.

The core engine is defined in:
> 📁 [`temporal_comp_gym/envs/sequential_colour/core.py`](../temporal_comp_gym/envs/sequential_colour/core.py)

---

## 1. Initialization and Configuration (`__init__`)

When the environment is instantiated, it configures the difficulty parameters and constructs the spatial arena.

### Observation and Action Spaces
This is inherently a **vision-based** environment. The observation space is an 84x84 RGB image array, making it explicitly designed for Convolutional Neural Networks (CNNs). The action space is discrete, allowing standard 4-way movement.

```python
# Action space: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
self.action_space = spaces.Discrete(4)

# Observation space: 84x84x3 RGB image
self.obs_size = 84
self.observation_space = spaces.Box(
    low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8
)
```

### Difficulty Constraints
The `__init__` method accepts several hyperparameters that drastically alter the task's temporal complexity:
*   `sequence_length_range`: Determines how many colors are shown during the memorisation phase.
*   `retention_delay_steps`: The duration of the sensory deprivation phase.
*   `allow_consecutive_duplicates`: E.g., allowing Red -> Red -> Blue.
*   `hard_death_mode`: If True, a single navigation mistake terminates the episode.
*   `infinite_mode`: If True, completing a sequence instantly starts a new one indefinitely instead of resetting.

---

## 2. Sequence Generation (`reset`)

When `reset()` is called, the environment generates a fresh target sequence and prepares the first temporal phase. 

```python
def reset(self, seed=None, options=None):
    # ... resets score and timers ...
    self.current_phase = "memorisation"
    
    seq_len = self.np_random.integers(self.sequence_length_range[0], self.sequence_length_range[1] + 1)
    self.sequence = []
    last_color_idx = -1
    
    # Generate the target sequence dynamically based on vocabulary size
    for _ in range(seq_len):
        if self.allow_consecutive_duplicates:
            idx = self.np_random.integers(0, self.colour_vocabulary_size)
        else:
            choices = [i for i in range(self.colour_vocabulary_size) if i != last_color_idx]
            idx = self.np_random.choice(choices)
            
        self.sequence.append(self.vocab_colors[idx])
        last_color_idx = idx
        
    self.current_seq_idx = 0
    # ...
```

---

## 3. The Three Temporal Phases (`_get_obs` and `step`)

The environment fundamentally revolves around three shifting phases. The `_get_obs` method dynamically changes the visual output based on the active phase.

### Phase 1: Memorisation
The environment flashes the target colors sequentially across the entire screen. Movement actions are ignored during this time.

```python
# From _get_obs()
if self.current_phase == "memorisation":
    # The entire 84x84 screen becomes the target color
    color = self.sequence[self.current_seq_idx]
    obs = np.full((self.obs_size, self.obs_size, 3), color, dtype=np.uint8)
```

In the `step()` function, a timer simply advances the sequence index until all colors have been shown, eventually triggering Phase 2.

### Phase 2: Retention
The screen goes completely black. The agent must hold the sequence in its internal memory (e.g., an LSTM or GRU state) while receiving zero sensory input.

```python
# From _get_obs()
elif self.current_phase == "retention":
    # Sensory deprivation (all zeros)
    obs = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
```

The `step()` function counts down `retention_delay_steps`. Once complete, it randomizes the room walls and triggers Phase 3.

### Phase 3: Navigation
The 84x84 visual observation shifts to a top-down view of the room. The agent (a white square) must physically move to the wall containing the current target color in the sequence.

```python
# From step() during navigation phase
self.agent.move(action)

agent_rect = self.agent.get_rect()
wall_rects = self.room.get_wall_rects()

hit_wall = None
for key, rect in wall_rects.items():
    if agent_rect.colliderect(rect): # Checks AABB overlap
        hit_wall = key
        break
```

---

## 4. Collision Logic and Rewards

When the agent successfully collides with a wall in Phase 3, the engine evaluates the choice against the memorized sequence.

**Correct Choice:**
```python
if touched_color == target_color:
    if self.reward_density == 'dense':
        reward += self.reward_correct_wall
    self.current_seq_idx += 1
    
    if self.current_seq_idx >= len(self.sequence):
        # Sequence complete!
        if self.reward_density == 'sparse':
            reward += self.reward_sequence_completion
            
        if self.infinite_mode:
            self.current_phase = "memorisation"
            # ... automatically generates next sequence loop ...
        else:
            terminated = True
    else:
        # Sequence not finished: Move to the next color in the sequence
        self.agent.reset()
        self._randomise_room() # Re-scrambles wall colors
```

**Incorrect Choice:**
```python
else:
    # Touched the wrong wall!
    self.score -= self.penalty_severity
    reward -= self.penalty_severity
    
    if self.hard_death_mode:
        terminated = True
    else:
        self.agent.reset() # Soft reset to try again
```
