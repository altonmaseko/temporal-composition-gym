# Package Architecture and Environment Registration

This document details the Python package structure for the `temporal_composition_gym` benchmark. It explains how the repository is laid out, how the custom environments are organized internally, how they are registered into the global Gymnasium environment registry, and what default settings they start with.

## Package Structure Overview

The repository is an installable Python package. The package name defined in `setup.py` is `temporal_composition_gym`, but the environment source code lives in the `temporal_comp_gym` directory.

> 📁 [`setup.py`](../setup.py)

```
temporal-composition-gym/
│
├── setup.py                          # Package configuration and environment dependencies
├── requirements.txt                  # Pinned protocol v2 environment (training and tests included)
├── pytest.ini                        # Test configuration (see 13_automated_test_suites.md)
├── temporal_comp_gym/                # Environment package
│   ├── __init__.py                   # Registers the three environments with Gymnasium
│   ├── oracles.py                    # Scripted privileged controllers (solvability checks only)
│   └── envs/
│       ├── __init__.py               # Exposes the three environment entry points
│       ├── hazardous_delivery/       # core.py, objects.py, wrappers.py, agent_observation_space.txt
│       ├── deferred_maintenance/     # core.py, objects.py, wrappers.py, agent_observation_space.txt, assets/
│       └── sequential_colour/        # core.py, objects.py, wrappers.py, agent_observation_space.txt
├── agents/                           # Training code for the baselines (see agents/README.md)
│   ├── common/                       # Plan, runner, env factory, seeding, evaluation, logging, replay
│   ├── reward_machines/              # Event-driven reward machines
│   ├── drqn/  qrm/  ppo_rm/          # Custom learners and their entry points
│   └── train_dqn.py  train_ppo.py  train_recurrent_ppo.py
├── run_scripts/baseline/             # Launchers, preflight, plan file, protocol and launch docs, results/
├── tests/                            # Automated tests; tests/visual/ holds display demos
├── play_human/                       # Keyboard-playable versions of the three tasks
└── gym_docs/                         # This documentation
```

`setup.py` uses `setuptools.find_packages()`, which finds every directory with an `__init__.py`: `temporal_comp_gym` and its `envs` subpackages, and also `agents`, `agents.common` and `agents.reward_machines`. Install the checkout in editable mode (`pip install -e .`) so that imports always use the code in this checkout. The training scripts and pytest also add the repository root to `sys.path` themselves, so they work from the repository root without an install.

## Exposing Environments (`envs/__init__.py`)

Each environment is implemented in its own subdirectory of `temporal_comp_gym/envs/`. To keep imports short, the three entry points are re-exported at the `envs` module level:

> 📁 [`temporal_comp_gym/envs/__init__.py`](../temporal_comp_gym/envs/__init__.py)

```python
from temporal_comp_gym.envs.hazardous_delivery import HazardousDeliveryEnv
from temporal_comp_gym.envs.deferred_maintenance import DeferredMaintenanceEnv
from temporal_comp_gym.envs.sequential_colour import SequentialColourEnv
```

So an environment can be referenced as `temporal_comp_gym.envs:HazardousDeliveryEnv` without knowing the inner file structure. The three entry points are not built the same way:

| Entry point | What it is | What `gym.make` returns inside its standard wrappers |
|---|---|---|
| `HazardousDeliveryEnv` | A function in `hazardous_delivery/__init__.py` | `HazardousDeliveryCoreEnv` wrapped in `HazardousDeliveryObservationWrapper` (flat vector) |
| `DeferredMaintenanceEnv` | A class in `deferred_maintenance/__init__.py` that subclasses `DeferredMaintenanceObservationWrapper` | The core `DeferredMaintenanceEnv` (from `core.py`) wrapped by that flattening wrapper (25-value vector) |
| `SequentialColourEnv` | The core class itself, from `sequential_colour/core.py` | The core environment, unwrapped (84×84 RGB frame) |

## Registering into the Gymnasium Registry (`__init__.py`)

Gymnasium uses a global registry to manage environments. Users create environments with `gym.make("EnvName-vX")`, so the custom environments must be registered first.

Registration happens in the package's top-level initialization file. Importing the package (`import temporal_comp_gym`) runs this code and adds the three environments to the registry:

> 📁 [`temporal_comp_gym/__init__.py`](../temporal_comp_gym/__init__.py)

```python
from gymnasium.envs.registration import register

register(
    id='TemporalComp/HazardousDelivery-v0',
    entry_point='temporal_comp_gym.envs:HazardousDeliveryEnv',
)

register(
    id='TemporalComp/DeferredMaintenance-v0',
    entry_point='temporal_comp_gym.envs:DeferredMaintenanceEnv',
)

register(
    id='TemporalComp/SequentialColour-v0',
    entry_point='temporal_comp_gym.envs:SequentialColourEnv',
)
```

### Understanding the Registration Parameters

1. **`id` (String)**: This is the unique identifier string used in `gym.make()`. It generally follows the format `Namespace/EnvName-vVersion`.
   * **Namespace (`TemporalComp/`)**: Groups related environments together, preventing naming collisions with other standard Gymnasium environments or external plugins.
   * **Environment Name**: The name of the specific task (`HazardousDelivery`, `DeferredMaintenance`, `SequentialColour`).
   * **Versioning (`-v0`)**: Gymnasium practice is to bump the version when an environment's behaviour changes. Protocol v2 changed the task rules (events, step limits, seeded and solvable layouts) but kept the `-v0` IDs. A `-v0` result from the old pilot code is therefore **not** comparable with a protocol v2 result; check the code revision recorded with a result.

2. **`entry_point` (String)**: This tells Gymnasium where to find the environment upon instantiation.
   * It uses the format `module.path:Name`.
   * For example, `temporal_comp_gym.envs:HazardousDeliveryEnv` tells Gymnasium to import `HazardousDeliveryEnv` from `temporal_comp_gym.envs` (exposed by `envs/__init__.py` as shown above) and call it with the keyword arguments given to `gym.make`.

No `max_episode_steps` is registered, so `gym.make` does not add a `TimeLimit` wrapper. Each environment enforces its own limit through its `max_steps` argument and reports `termination_cause="timeout"`. Do not pass `max_episode_steps` to `gym.make`: Gymnasium would consume it and add its own `TimeLimit`, bypassing the environment's limit and its timeout cause. Use `max_steps` instead.

## Default Settings

Any keyword argument of an environment's constructor can be passed to `gym.make`. The code defaults are documented in docs 04, 07 and 10. The protocol v2 baselines do not rely on them: [`agents/common/env_factory.py`](../agents/common/env_factory.py) passes the frozen main-condition settings explicitly. Where they differ:

| Environment | Plain `gym.make` default | Protocol v2 main condition (`env_factory.make_env`) |
|---|---|---|
| Hazardous Delivery | Delivered count visible: 9-value observation | `include_inventory_in_state=False`: 8 values (decision D1) |
| Deferred Maintenance | `distractor_stations=False`; unscaled 25-value observation | `distractor_stations=True` (decision D2); `MaintenanceScaledObservation` added on top: 28 values |
| Sequential Colour | Same as the main condition | Same as the code defaults |

The step limits are the same in both columns: 500 (delivery), 4,000 (maintenance) and 300 (colour) steps.

## Usage Example

Once the package is installed (or you are running from the repository root), import it to trigger the registration, then use `gym.make()` as usual:

```python
import gymnasium as gym
import temporal_comp_gym  # This import triggers the registration in __init__.py

# Gymnasium now knows how to find and initialize the custom environment
env = gym.make('TemporalComp/HazardousDelivery-v0')

# Begin the standard RL loop
obs, info = env.reset(seed=0)
# ...
```

To get exactly the environment the protocol v2 baselines use:

```python
from agents.common.env_factory import make_env

env = make_env("delivery")        # or "colour", "maintenance"
obs, info = env.reset(seed=0)
```
