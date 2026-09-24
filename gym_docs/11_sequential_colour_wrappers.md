# Sequential Colour Environment: Observation Space and Wrappers

Unlike state-vector environments designed for tabular reinforcement learning, the **Sequential Colour** task is fundamentally a visual challenge. The agent receives no explicit internal state variables—no spatial coordinates, no phase indicators, and no target sequence strings. It must derive all task context entirely from a continuous stream of pixels.

The details of the visual observation space and its modifications are governed by:
> 📁 [`temporal_comp_gym/envs/sequential_colour/agent_observation_space.txt`](../temporal_comp_gym/envs/sequential_colour/agent_observation_space.txt)
> 📁 [`temporal_comp_gym/envs/sequential_colour/wrappers.py`](../temporal_comp_gym/envs/sequential_colour/wrappers.py)

---

## 1. The Raw Observation Space

The raw observation passed from the core engine to the agent is a 3D NumPy array representing an RGB image frame. It is explicitly sized to be compatible with standard Convolutional Neural Network (CNN) architectures.

*   **Type**: `gym.spaces.Box`
*   **Shape**: `(84, 84, 3)`
*   **Data Type**: `np.uint8` (Integer values from 0 to 255)
*   **Format**: RGB (Red, Green, Blue)

### What the Agent Sees (By Phase)
Because this task revolves around temporal composition, the exact meaning and layout of the 84x84 RGB array shifts dynamically depending on the current temporal phase of the episode:

**Phase 1: Memorisation**
*   **Visual**: A completely solid-colored 84x84x3 image.
*   **Context**: Represents a single color from the sequence. The agent must process this solid color visually and push it into its recurrent memory (e.g., LSTM/GRU).

**Phase 2: Retention (Delay)**
*   **Visual**: A completely solid black image (all RGB values are strictly `0`).
*   **Context**: Acts as a sensory deprivation buffer. The agent must successfully retain the memorized sequence without any visual cues or prompts.

**Phase 3: Navigation**
*   **Visual**: A top-down visual representation of the physical room, scaled precisely to fit the 84x84 window.
*   **Context**: The agent (rendered as a white square) must use this spatial view to identify the four coloured boundaries and physically navigate towards the correct target wall.

---

## 2. The Observation Wrapper (`GaussianNoiseWrapper`)

To evaluate the robustness of the agent's visual feature extractor and its ability to maintain memory under imperfect, noisy conditions, the raw image can be modified using the `GaussianNoiseWrapper`.

### Wrapper Implementation
When initialized with an `observation_noise_level > 0.0`, the wrapper intercepts the clean 84x84x3 frame and injects random Gaussian noise before passing the tensor to the agent.

```python
import gymnasium as gym
import numpy as np

class GaussianNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, observation_noise_level=0.0):
        super().__init__(env)
        self.noise_level = observation_noise_level
        
    def observation(self, obs):
        if self.noise_level > 0.0:
            # 1. Generate Gaussian noise scaled to the maximum pixel range (255)
            noise = np.random.normal(0, self.noise_level * 255, obs.shape)
            
            # 2. Add noise to the clean observation
            # 3. Clip the resulting values to ensure they remain valid 8-bit RGB integers
            noisy_obs = np.clip(obs + noise, 0, 255).astype(np.uint8)
            
            return noisy_obs
            
        # Return clean observation if noise is explicitly disabled
        return obs
```

### Noise Logic Breakdown
1.  **`np.random.normal`**: Generates a matrix of random noise centered at 0. The spread (standard deviation) is calculated by multiplying the `noise_level` scalar by the maximum pixel value `255`.
2.  **`obs + noise`**: The random noise matrix is added pixel-by-pixel to the clean RGB frame, distorting the image.
3.  **`np.clip(..., 0, 255)`**: Because adding random noise can push pixel values below 0 or above 255 (which would crash the `np.uint8` conversion or cause severe graphical artifacting due to integer overflow), the tensor is strictly clamped back to the valid 8-bit color range.
