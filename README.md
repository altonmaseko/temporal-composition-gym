# Temporal Composition Gym

A benchmark suite of non-Markovian reinforcement learning environments designed to evaluate temporal composition and memory-augmented agents. The suite spans tabular, continuous-control, and image-based settings to test how agents handle delayed, history-dependent rewards across varying sensory modalities.

---

## Overview

This repository contains three distinct environments built using the `Gymnasium` API. Standard Markovian algorithms are mathematically expected to plateau in these environments, requiring the integration of implicit sequence models (e.g., LSTMs) or explicit automaton-based reward representations (e.g., Reward Machines) to solve them successfully.

### Environments

#### 1. The Hazardous Delivery Task (Tabular Setting)
* **ID:** `TemporalComp/HazardousDelivery-v0`
* **Engine:** MiniGrid
* **Description:** A discrete grid-based puzzle. The agent navigates a 2D grid containing hazards (Death Squares and Damage Squares) to collect and deliver packages to specific destinations. The layout changes every episode to prevent route memorization.
* **Observation Space:** A 1D discrete integer array. It includes coordinate position, a 4-value local surroundings sensor, and current square status. Global mapping is completely hidden to enforce memory reliance.
* **Action Space:** Discrete (Up, Down, Left, Right).

#### 2. The Deferred Maintenance Task (Continuous-Control Setting)
* **ID:** `TemporalComp/DeferredMaintenance-v0`
* **Engine:** PyMunk & Pygame
* **Description:** A 2D top-down physics simulation. A rocket, subject to constant downward gravity, must navigate a winding path of safe zones. The agent collects "Maintenance Codes" and must retain them in memory to service corresponding colored landing pads later in the stage. Stages can be hardcoded via 2D arrays or procedurally generated.
* **Observation Space:** A flat 1D vector containing kinematics, micro-navigation sensors, waypoints, floor codes, and station radar distances.
* **Action Space:** Continuous (Thrust and Angular Momentum).

#### 3. The Sequential Colour Navigation Task (Image-Based Setting)
* **ID:** `TemporalComp/SequentialColour-v0`
* **Engine:** Custom Pygame
* **Description:** A task split into two distinct temporal phases. During the memorisation phase, the agent observes a sequence of solid-colored image frames. During the navigation phase, the agent is placed in a rectangular room and must touch the walls matching the memorised color sequence in the exact order.
* **Observation Space:** 3D NumPy array of shape `(84, 84, 3)` representing an RGB image frame. 
* **Action Space:** Discrete spatial movements (Up, Down, Left, Right).

---

## Architecture and UI Features

* **Modular Design:** Each environment is separated into `core.py`, `objects.py`, and `wrappers.py` for easy manipulation of state spaces and rendering logic.
* **Custom Rendering:** The Pygame UI features a split-screen layout for human rendering, including dynamic stats, active memory trackers, and a custom legend. 
* **Configurable Difficulty:** Environments include exposed parameters to scale cognitive load, such as sequence lengths, temporal delays, grid sizes, and distractor stimuli.

---

## Installation

To install the package locally for development and training, clone the repository and run the following in the root directory:

```bash
pip install -e .

```

### Dependencies

- `gymnasium`
- `minigrid`
- `pygame`
- `pymunk`
- `numpy`

## Usage

Once installed, the environments can be instantiated using standard Gymnasium syntax:

```python
import gymnasium as gym
import temporal_comp_gym

env = gym.make(
    "TemporalComp/DeferredMaintenance-v0",
    render_mode="human"
)

obs, info = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```
