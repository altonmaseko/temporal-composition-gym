# Package Architecture and Environment Registration

This document details the Python package structure for the `temporal_composition_gym` benchmark. It explains how the package is structured, how the custom environments are organized internally, and the mechanism by which they are exposed and registered into the global Gymnasium environment registry.

## Package Structure Overview

The repository is structured as a standard installable Python package. Although the package name defined in `setup.py` is `temporal_composition_gym`, the actual source code is housed within the `temporal_comp_gym` directory.

> 📁 [`setup.py`](../setup.py)

A typical structure looks like this:
```
temporal-composition-gym/
│
├── setup.py                          # Package configuration and dependencies
└── temporal_comp_gym/                # Root Python module
    ├── __init__.py                   # Top-level init; registers environments with Gymnasium
    └── envs/                         # Directory containing all environment definitions
        ├── __init__.py               # Exposes environment classes at the module level
        ├── hazardous_delivery/       # Implementation of the Hazardous Delivery env
        ├── deferred_maintenance/     # Implementation of the Deferred Maintenance env
        └── sequential_colour/        # Implementation of the Sequential Colour env
```

The `setup.py` uses `setuptools.find_packages()` to automatically discover the `temporal_comp_gym` module and its submodules (like `envs`), making them available when the package is installed via pip (e.g., `pip install -e .`).

## Exposing Environments (`envs/__init__.py`)

To keep the codebase modular, each custom environment is implemented in its own subdirectory within the `temporal_comp_gym/envs/` module. However, to make these environments easily importable and to clean up the namespaces, they are hoisted to the `envs` module level.

This is achieved in the environment module's initialization file:

> 📁 [`temporal_comp_gym/envs/__init__.py`](../temporal_comp_gym/envs/__init__.py)

```python
from temporal_comp_gym.envs.hazardous_delivery import HazardousDeliveryEnv
from temporal_comp_gym.envs.deferred_maintenance import DeferredMaintenanceEnv
from temporal_comp_gym.envs.sequential_colour import SequentialColourEnv
```

By doing this, an environment class can be referenced simply as `temporal_comp_gym.envs:HazardousDeliveryEnv` rather than forcing the importer to know the exact inner file structure (e.g., `temporal_comp_gym.envs.hazardous_delivery.core:HazardousDeliveryEnv`).

## Registering into the Gymnasium Registry (`__init__.py`)

Gymnasium uses a global registry to manage environments. Instead of instantiating environment classes manually, users rely on the `gym.make("EnvName-vX")` API. For this to work, custom environments must be registered with Gymnasium.

This registration happens in the top-level initialization file. Whenever a user imports the package (`import temporal_comp_gym`), this code runs automatically, injecting the custom environments into the Gymnasium registry.

> 📁 [`temporal_comp_gym/__init__.py`](../temporal_comp_gym/__init__.py)

Here is how the registration is implemented:

```python
from gymnasium.envs.registration import register

register(
    id='TemporalComp/HazardousDelivery-v0',
    entry_point='temporal_comp_gym.envs:HazardousDeliveryEnv',
)

# Registration repeats similarly for DeferredMaintenance-v0 and SequentialColour-v0
```

### Understanding the Registration Parameters

1. **`id` (String)**: This is the unique identifier string used in `gym.make()`. It generally follows the format `Namespace/EnvName-vVersion`.
   * **Namespace (`TemporalComp/`)**: Groups related environments together, preventing naming collisions with other standard Gymnasium environments or external plugins.
   * **Environment Name**: The name of the specific task (e.g., `HazardousDelivery`, `DeferredMaintenance`, `SequentialColour`).
   * **Versioning (`-v0`)**: It is a standard Gymnasium practice to append a version number. If the environment logic, observations, or action spaces fundamentally change in the future, developers can release a `-v1` without breaking backward compatibility for algorithms trained on `-v0`.

2. **`entry_point` (String)**: This tells Gymnasium exactly where to find the environment class upon instantiation. 
   * It uses the format `module.path:ClassName`.
   * For example, `temporal_comp_gym.envs:HazardousDeliveryEnv` instructs Gymnasium to import `HazardousDeliveryEnv` from the `temporal_comp_gym.envs` module (which, as explained above, was helpfully exposed in `envs/__init__.py`).

## Usage Example

Because of this architectural setup, utilizing the custom environments in a reinforcement learning script is completely seamless. Once the package is installed, you only need to import it to trigger the registration, and then you can use `gym.make()` as usual:

```python
import gymnasium as gym
import temporal_comp_gym  # This import triggers the registration in __init__.py

# Gymnasium now knows how to find and initialize the custom environment!
env = gym.make('TemporalComp/HazardousDelivery-v0')

# Begin the standard RL loop
obs, info = env.reset()
# ...
```
