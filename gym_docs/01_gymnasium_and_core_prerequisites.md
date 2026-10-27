# Core Prerequisites and Foundational Libraries

The `temporal_composition_gym` benchmark is built on top of several foundational libraries that provide the infrastructure for reinforcement learning environments, physics simulation, and visual rendering.

## Foundational Libraries

There are two dependency lists:

*   **[`setup.py`](../setup.py)** lists what the environments themselves need (unpinned): `gymnasium`, `minigrid`, `pygame`, `pymunk`, `numpy`, `pytablericons`, `Pillow` and `requests`.
*   **[`requirements.txt`](../requirements.txt)** pins the complete protocol v2 environment, including the training and testing stack. It was taken from the `temporal_gym` conda environment (Python 3.10). Install from it on every machine that runs experiments, so all machines use the same versions.

### Environment libraries

*   **Gymnasium** (1.3.0): The standard API for reinforcement learning environments. It provides the core `Env` class our environments inherit from and the `reset`/`step`/`render` interface agents use.
*   **Minigrid** (3.1.0): A lightweight gridworld library built on Gymnasium. Hazardous Delivery is built on its grid, objects and rendering.
*   **Pygame** (2.6.1): Window rendering, keyboard input for the human-play scripts, and drawing. Sequential Colour also uses it to draw the agent's observation frame. `pygame-ce` must not be installed alongside it, because both use the `pygame` import name.
*   **Pymunk** (7.3.0): A 2D physics library, used for the rocket and level in Deferred Maintenance.
*   **NumPy** (2.2.6): Arrays for observations, actions and all numerical work, and the seeded random generators.
*   **Pillow** (12.3.0): Image handling for icons.
*   **PyTablerIcons** (1.0.1): The icon set used when rendering the environments for humans.
*   **Requests**: Fetches icon images when an icon URL is configured in an environment's `ICON_URLS` (empty by default).

### Training, logging and testing libraries (pinned in `requirements.txt` only)

*   **PyTorch** (2.14.0): The custom DRQN and QRM learners, and the backend of Stable-Baselines3.
*   **Stable-Baselines3** (2.9.0) and **sb3-contrib** (2.9.0): DQN, PPO and Recurrent PPO.
*   **Weights & Biases** (`wandb` 0.30.0): Experiment tracking. Authentication is through `wandb login` only; the code never reads an API key.
*   **psutil**: Records RAM, disk and memory use in run metadata.
*   **pytest** (9.1.1): The automated test suite (see [`13_automated_test_suites.md`](13_automated_test_suites.md)).

On the development PC, always use the project interpreter (see `AGENTS.md`):

```bash
D:\miniconda3\envs\temporal_gym\python.exe -m pip install -r requirements.txt
D:\miniconda3\envs\temporal_gym\python.exe -m pip install -e .
```

## The Standard Gymnasium Reinforcement Learning Loop

To understand how `temporal_composition_gym` environments work, it is essential to understand the standard Gymnasium API. Every environment in this benchmark implements this standard interface, ensuring compatibility with most modern RL algorithms and frameworks.

### 1. Environments and Spaces

An **Environment** encapsulates the rules, dynamics, and state of the world. Agents do not interact with the environment's internal state directly; instead, they observe it through an `Observation Space` and act on it through an `Action Space`.

*   **Action Space** (`env.action_space`): Defines the valid actions an agent can take. Here: 4 discrete moves (up, down, left, right) in Hazardous Delivery and Sequential Colour, and 2 continuous values (thrust, rotation) in Deferred Maintenance.
*   **Observation Space** (`env.observation_space`): Defines the format and bounds of the observations the agent receives. Here: a short vector (Hazardous Delivery, Deferred Maintenance) or an 84×84 RGB image (Sequential Colour).

### 2. The RL Loop (`reset` and `step`)

The interaction between an agent and the environment follows a continuous loop:

1.  **Initialize**: The environment is initialized to a starting state using `reset()`. Passing `seed=` makes the episode reproducible.
2.  **Act and Observe**: The agent chooses an action, and the environment advances by one timestep using `step(action)`.
3.  **Feedback**: `step()` returns the new observation, the reward, whether the episode has terminated or truncated, and an info dictionary.

*   **Terminated** means the task itself ended the episode (success or failure).
*   **Truncated** means the episode hit its time limit.

All three environments enforce their own step limit internally (`max_steps`: 500, 300 and 4,000 steps), and the two flags are never both `True`.

Here is the loop with one of this benchmark's environments:

```python
import gymnasium as gym
import temporal_comp_gym  # registers the TemporalComp/... environments

# 1. Create the environment
env = gym.make("TemporalComp/HazardousDelivery-v0")

# 2. Reset the environment to start a new episode
# Returns the initial observation and an info dictionary
observation, info = env.reset(seed=42)

total_reward = 0
done = False

# 3. The main interaction loop
while not done:
    # Sample a random action from the environment's action space
    # (In a real setup, your trained agent model would select this action)
    action = env.action_space.sample()
    
    # Take the step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    
    # An episode ends if it either reaches a terminal state (terminated)
    # or hits the environment's step limit (truncated)
    done = terminated or truncated

print(f"Episode finished with total reward: {total_reward}")
print(info["termination_cause"], info["is_success"])

# 4. Clean up
env.close()
```

When you examine the implementations of the environments in `temporal_composition_gym`, you will see this exact structure. The physics and temporal rules are embedded inside the custom `reset()` and `step()` methods of the environment classes.

### 3. The `info` Dictionary in This Benchmark

Every `reset()` and `step()` of all three environments returns `info["events"]`: a list of `(name, argument)` pairs describing what happened on that transition (for example `("pickup", None)` or `("wall_touch", 2)`). The last step of an episode also carries `info["termination_cause"]` and `info["is_success"]`. The reward machines used by the explicit-memory agents read only these events. With `debug_info=True` an environment also adds `info["debug"]` with hidden task state; this is for tests and scripted oracles only, never for training. Each environment's events are listed in its `agent_observation_space.txt` and in docs 04, 07 and 10.

## Environment Configurations and Display Adjustments

The project's developer notes ([`notes.txt`](../notes.txt)) give a few practical tips. They are summarised here, corrected where the code has changed:

### Game Parameters
Each environment's parameters are the keyword arguments of its main class's `__init__` in its `core.py`:

> 📁 `temporal_comp_gym/envs/<env_name>/core.py`

They can be passed to `gym.make(...)`, for example `gym.make("TemporalComp/HazardousDelivery-v0", num_packages=2)`. The protocol v2 experiments do not use the code defaults implicitly: [`agents/common/env_factory.py`](../agents/common/env_factory.py) passes the frozen main-condition settings explicitly, and those differ from the code defaults in places (see [`02_package_architecture_and_registration.md`](02_package_architecture_and_registration.md)). Change settings for your own experiments through `gym.make` arguments rather than by editing the defaults in `core.py`, which would change the baseline configuration.

### Adjusting Visual Display Speed
The files in `tests/` are now automated checks with no display. The visual demos that open a window and play random actions live in `tests/visual/` (`demo_<env_name>.py`); pytest does not collect them.

> 📁 [`tests/visual/`](../tests/visual/)
> *Tip: Adjust the `time.sleep(x)` value in the loop of the respective `demo_<env_name>.py` file to slow down or speed up the rendering.*

### Playing the Games Manually
To better understand the task constraints and physics of a particular environment, you can play the games yourself.
*   **Tip:** Run the human-playable scripts in the `play_human` directory (see [`12_human_interaction_loops.md`](12_human_interaction_loops.md)). This is a good way to get an intuitive feel for the tasks before training an agent.
