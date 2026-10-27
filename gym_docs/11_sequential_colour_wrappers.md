# Sequential Colour Environment: Observation Space and Wrappers

Unlike the state-vector environments, the **Sequential Colour** task is a visual challenge. The agent receives no explicit internal state variables: no spatial coordinates, no phase indicator and no target sequence. It must derive all task context from a stream of pixels.

The details of the visual observation space and its modifications are governed by:
> 📁 [`temporal_comp_gym/envs/sequential_colour/agent_observation_space.txt`](../temporal_comp_gym/envs/sequential_colour/agent_observation_space.txt)
> 📁 [`temporal_comp_gym/envs/sequential_colour/wrappers.py`](../temporal_comp_gym/envs/sequential_colour/wrappers.py)

---

## What Changed in Protocol v2

*   **The observation itself did not change.** It is still the raw 84×84 RGB frame, and the baselines receive it unwrapped (no noise, no resizing, no greyscale).
*   **The noise wrapper now uses its own seeded generator.** Before, `GaussianNoiseWrapper` drew noise from NumPy's global random state, so noise was not reproducible and could interfere with other code using the global state. Adding noise now never changes the sequence or the rooms.
*   **`info` changed, not the observation.** Every step now reports `events`, and the last step reports `termination_cause` and `is_success`; see [`10_sequential_colour_core.md`](10_sequential_colour_core.md). The target colour is never in `info` during training or evaluation.

---

## 1. The Raw Observation Space

The observation produced by the core engine is a 3D NumPy array representing an RGB image. It is sized to be compatible with standard Convolutional Neural Network (CNN) architectures.

*   **Type**: `gym.spaces.Box`
*   **Shape**: `(84, 84, 3)`
*   **Data Type**: `np.uint8` (Integer values from 0 to 255)
*   **Format**: RGB (Red, Green, Blue). The baselines keep all three channels: the task is decided by colour, and converting to grey would make colours harder to tell apart.

`gym.make("TemporalComp/SequentialColour-v0")` returns the core engine with only Gymnasium's standard `PassiveEnvChecker` and `OrderEnforcing` wrappers. No observation wrapper is applied, and the baseline factory (`agents/common/env_factory.py`) adds none either.

### What the Agent Sees (By Phase)
The meaning of the 84×84 RGB array depends on the current phase:

**Phase 1: Memorisation**
*   **Visual**: A solid-coloured 84×84×3 image in the current cue colour.
*   **Timing**: Each colour stays for `frames_per_colour` observations (default 5), starting with the reset observation. Two identical colours in a row (only possible with `allow_consecutive_duplicates=True`) look like one long cue; the agent can only tell them apart by counting frames.
*   **Context**: The agent must store each colour in its recurrent memory (e.g., LSTM/GRU). Actions have no effect.

**Phase 2: Retention (Delay)**
*   **Visual**: A solid black image (all values `0`).
*   **Timing**: `retention_delay_steps` observations (default 2).
*   **Context**: The agent must hold the sequence with no visual input. Actions have no effect.

**Phase 3: Navigation**
*   **Visual**: A top-down view of the room scaled to fit the 84×84 frame: black background, four coloured walls about 5 pixels thick, and the agent as a white square of about 5 pixels in the centre.
*   **Context**: The agent must move to the wall showing the current target colour. After a correct, non-final wall the agent is re-centred in a room with new wall colours; after a wrong wall it is re-centred in the same room. Nothing in the frame says which colour is the target or how far through the sequence the agent is.

---

## 2. The Observation Wrapper (`GaussianNoiseWrapper`)

To evaluate the robustness of an agent's visual features and memory under noisy conditions, the image can be distorted with the `GaussianNoiseWrapper`. It is not part of the protocol v2 baselines and is never applied automatically; you wrap the environment yourself:

```python
from temporal_comp_gym.envs.sequential_colour import GaussianNoiseWrapper
env = GaussianNoiseWrapper(gym.make("TemporalComp/SequentialColour-v0"), observation_noise_level=0.1)
```

### Wrapper Implementation
When initialized with an `observation_noise_level > 0.0`, the wrapper adds random Gaussian noise to every observation (in all phases) before passing it to the agent.

```python
class GaussianNoiseWrapper(gym.ObservationWrapper):
    """Adds pixel noise from the wrapper's own seeded generator.

    The generator is independent of the environment's RNG, so adding noise never changes
    sequences or room layouts. It is reseeded whenever reset() receives a seed.
    """

    def __init__(self, env, observation_noise_level=0.0):
        super().__init__(env)
        self.noise_level = observation_noise_level
        self._noise_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._noise_rng = np.random.default_rng([int(seed), 0x5EED])
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        if self.noise_level > 0.0:
            noise = self._noise_rng.normal(0, self.noise_level * 255, obs.shape)
            noisy_obs = np.clip(obs + noise, 0, 255).astype(np.uint8)
            return noisy_obs
        return obs
```

### Noise Logic Breakdown
1.  **Own generator**: The wrapper keeps a private NumPy generator. `reset(seed=s)` reseeds it from `s`, so the same seed gives the same noise. Before the first seeded reset (or if reset is never given a seed) it starts from an unseeded generator, so the noise is not reproducible in that case.
2.  **`normal(0, noise_level × 255)`**: Generates one noise value per pixel and channel, centred at 0, with a standard deviation of `noise_level × 255`.
3.  **`obs + noise`**: The noise is added to the clean RGB frame.
4.  **`np.clip(..., 0, 255)`**: The result is clamped to the valid 8-bit range before converting back to `np.uint8`, so values never wrap around.

At `observation_noise_level = 0.0` (the default) the wrapper returns the frame unchanged.
