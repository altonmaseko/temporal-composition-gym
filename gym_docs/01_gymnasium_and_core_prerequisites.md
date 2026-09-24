# Core Prerequisites and Foundational Libraries

The `temporal_composition_gym` benchmark is built on top of several foundational libraries that provide the infrastructure for reinforcement learning environments, physics simulation, and visual rendering. 

## Foundational Libraries

As listed in `setup.py`, the core dependencies include:

*   **Gymnasium**: The standard API for reinforcement learning environments. It provides the core `Env` class that our environments inherit from, defining the standard interface (`reset`, `step`, `render`) that RL agents use to interact with the environment.
*   **Minigrid**: A lightweight, fast gridworld environment library built on top of Gymnasium. It provides a flexible grid system, objects, and rendering utilities that we leverage for our grid-based tasks.
*   **Pygame**: A set of Python modules designed for writing video games. In this benchmark, `pygame` is used for handling window rendering, processing user inputs (like keyboard events when playing manually), and drawing graphics efficiently.
*   **Pymunk**: A 2D physics library for Python. It is used to simulate complex, realistic physical interactions (like collisions, gravity, and kinematics) within the continuous-space environments.
*   **NumPy**: The fundamental package for scientific computing in Python. It is extensively used for representing states, observations, actions, and performing fast numerical operations across the benchmark.
*   **Pillow**: A Python Imaging Library (PIL) fork, used for image processing and handling visual assets.
*   **PyTablerIcons**: A library providing access to Tabler Icons, used for rendering specific visual elements or icons within the environments.

## The Standard Gymnasium Reinforcement Learning Loop

To understand how `temporal_composition_gym` environments work, it is essential to understand the standard Gymnasium API. Every environment in this benchmark implements this standard interface, ensuring compatibility with most modern RL algorithms and frameworks.

### 1. Environments and Spaces

An **Environment** encapsulates the rules, dynamics, and state of the world. Agents do not interact with the environment's internal state directly; instead, they observe it through an `Observation Space` and act on it through an `Action Space`.

*   **Action Space** (`env.action_space`): Defines the valid actions an agent can take. For example, a discrete space of 4 actions (up, down, left, right) or a continuous space for applying forces in specific directions.
*   **Observation Space** (`env.observation_space`): Defines the format and bounds of the observations the agent receives. This could be a 1D array of internal state variables or a 3D array representing pixel data.

### 2. The RL Loop (`reset` and `step`)

The interaction between an agent and the environment follows a continuous loop:

1.  **Initialize**: The environment is initialized to a starting state using `reset()`.
2.  **Act and Observe**: The agent chooses an action, and the environment advances by one timestep using `step(action)`. 
3.  **Feedback**: `step()` returns the new observation, the reward, whether the episode has terminated or truncated, and an additional info dictionary.

Here is a concrete code example of this standard loop in action:

```python
import gymnasium as gym
import numpy as np

# 1. Create the environment
# (In practice, you would initialize one of the temporal_composition_gym environments)
env = gym.make("CartPole-v1")

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
    # or hits a predetermined time limit (truncated)
    done = terminated or truncated

print(f"Episode finished with total reward: {total_reward}")

# 4. Clean up
env.close()
```

When you examine the implementations of the environments in `temporal_composition_gym`, you will see this exact structure. The complex physics and temporal rules are embedded inside the custom `reset()` and `step()` methods of our environment classes.

## Environment Configurations and Display Adjustments

Based on the project's developer notes, there are a few important details to keep in mind when modifying or interacting with the environments:

### Game Parameters
The core logic and parameters for each specific game or environment are encapsulated within the `__init__` method of its main class. You can find these environment definitions in the `core.py` file located within each respective environment's directory. 

> 📁 `temporal_comp_gym/envs/<env_name>/core.py`
> *Tip: If you need to change the difficulty, grid size, physics parameters, or reward scaling, look for the `__init__` method in the corresponding environment's core file.*

### Adjusting Visual Display Speed
When running tests or observing a trained agent visually, the display rendering speed might be too fast to see what is happening clearly. You can adjust the frame rate manually during tests.

> 📁 [`tests/test_<env_name>.py`](../tests/)
> *Tip: Adjust the `time.sleep(x)` value within the main test loop of the respective test file to slow down or speed up the visual rendering.*

### Playing the Games Manually
To better understand the task constraints and physics of a particular environment, you can play the games yourself. 
*   **Tip:** Run the human-playable scripts located in the `play_human` directory for hands-on visual testing. This is an excellent way to debug logic and get an intuitive feel for the temporal composition tasks before training an agent.
