# Temporal Composition Gym

A benchmark suite of non-Markovian reinforcement learning environments for evaluating temporal composition and memory-augmented agents. The three tasks span a gridworld, continuous control and image-based settings. In each, the reward depends on what happened earlier in the episode, not only on the current observation.

## This branch: `new_experiments` (protocol v2)

This branch corrects the experiment pipeline and reruns the experiments, starting with the baselines. It began from [`main` at commit `972192a`](https://github.com/altonmaseko/temporal-composition-gym/tree/972192af99bf2cfd18bd7411047eec08b0f0f1b8).

*   **Done:** the task rules, reward machines and agents have been corrected (protocol v2), and the baseline training pipeline has been rebuilt and validated locally. The frozen protocol is [`run_scripts/baseline/BASELINE_PROTOCOL.md`](run_scripts/baseline/BASELINE_PROTOCOL.md), and the task changes are summarised in [`run_scripts/baseline/TASK_CHANGES_v2.md`](run_scripts/baseline/TASK_CHANGES_v2.md).
*   **Not redone yet:** hyperparameter tuning and stress testing. The old pilot tuning and stress-testing scripts were removed from this branch because they do not work with the protocol v2 code. New ones will be written in later phases.
*   **Historical material:** the old pilot results, exports and Optuna databases, and the Phase Stories describing them, are on `main` and in Git history only. They were produced by the pilot code, before the corrections, and are **historical, not validated results**. The audit utilities and source snapshots under `output/analysis/baseline_audit/` are kept as reference material.

---

## The Environments

All three use the Gymnasium API, emit an `info["events"]` list describing each transition, and end every episode with `info["termination_cause"]` and `info["is_success"]`. Each enforces its own step limit (truncation). Full details are in [`gym_docs/`](gym_docs/).

### 1. Hazardous Delivery (gridworld)
* **ID:** `TemporalComp/HazardousDelivery-v0` · **Engine:** MiniGrid · **Limit:** 500 steps
* **Task:** On a 10×10 grid with lava (instant death) and damage squares, pick up 3 packages one at a time, drop each at a destination, then enter the goal. Entering the goal early fails the episode. Layouts are random but seeded, and always solvable.
* **Observation:** A float vector. In the protocol v2 main condition it has 8 values: x, y, the four neighbouring cells, the current cell, and health. The number of packages delivered and carried is hidden, so the agent must remember it.
* **Actions:** Discrete (up, down, left, right).

### 2. Deferred Maintenance (continuous control)
* **ID:** `TemporalComp/DeferredMaintenance-v0` · **Engine:** Pymunk and Pygame · **Limit:** 4,000 steps
* **Task:** Fly a rocket under gravity along a winding path through 4 stages. Collect the maintenance codes in the early stages and remember which ones you hold. Once a "readiness" signal turns on in a later stage, hover over the matching station to service each code. Then leave the last stage holding nothing. Stages are generated from a seeded schedule that is always solvable.
* **Observation:** A vector of kinematics, a 3×3 local cell map, a waypoint to the next path cell, the floor code, readiness and a station radar. The baselines receive a fixed-scaled 28-value version.
* **Actions:** Continuous (thrust, rotation).

### 3. Sequential Colour (image-based)
* **ID:** `TemporalComp/SequentialColour-v0` · **Engine:** Pygame · **Limit:** 300 steps
* **Task:** Three phases. The screen first shows a sequence of 3–5 colours, then goes black for a short delay. Then, in a room with four coloured walls, the agent must touch the walls in the order of the sequence. The wall colours are reshuffled after each correct wall.
* **Observation:** An `(84, 84, 3)` RGB frame.
* **Actions:** Discrete (up, down, left, right).

---

## Installation

The pinned environment is in [`requirements.txt`](requirements.txt) (Python 3.10). `pip install -e .` installs the package itself.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

On the development PC, always use the project interpreter (see [`AGENTS.md`](AGENTS.md)):

```bash
D:\miniconda3\envs\temporal_gym\python.exe -m pip install -r requirements.txt
D:\miniconda3\envs\temporal_gym\python.exe -m pip install -e .
```

## Usage

```python
import gymnasium as gym
import temporal_comp_gym  # registers the environments

env = gym.make("TemporalComp/DeferredMaintenance-v0", render_mode="human")
obs, info = env.reset(seed=0)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

A plain `gym.make` uses each environment's code defaults. To get exactly what the protocol v2 baselines see, use `agents.common.env_factory.make_env("delivery" | "colour" | "maintenance")`; see [`gym_docs/02_package_architecture_and_registration.md`](gym_docs/02_package_architecture_and_registration.md). The `play_human/` scripts let you play the tasks with the keyboard (see [`gym_docs/12_human_interaction_loops.md`](gym_docs/12_human_interaction_loops.md)).

## Running the Tests

```bash
D:\miniconda3\envs\temporal_gym\python.exe -m pytest                  # full suite
D:\miniconda3\envs\temporal_gym\python.exe -m pytest -m "not slow"    # skip the slow checks
```

The suite checks the task rules, reward machines, learners, seeding and the training pipeline. It opens no windows, never contacts W&B, and writes only to temporary directories. See [`gym_docs/13_automated_test_suites.md`](gym_docs/13_automated_test_suites.md).

## Baseline Experiments

The baseline matrix is 55 runs on three machines, one per memory family. Seeds 1–5 throughout.

| Machine | Runs |
|---|---|
| memoryless | DQN (delivery, colour), PPO (maintenance), then the DQN-TS task-state controls (delivery, colour) |
| implicit | DRQN (delivery, colour), Recurrent PPO (maintenance) |
| explicit | QRM (delivery), QRM-Cue (colour), PPO-RM (maintenance) |

Machine setup, the cloud preflight, the exact launch command for each machine, restarting failed runs and collecting results are in **[`run_scripts/baseline/PRODUCTION_LAUNCH_v2.md`](run_scripts/baseline/PRODUCTION_LAUNCH_v2.md)** and [`run_scripts/baseline/README.md`](run_scripts/baseline/README.md). The agents and their configuration are described in [`agents/README.md`](agents/README.md).

**Results** go to `run_scripts/baseline/results/`, in one new directory per run:

```
run_scripts/baseline/results/v2/<full|smoke>/<env>/<algo>/seed<s>__<cfghash10>__<utc>-<rand>/
```

The local CSV and JSON files in these directories are the official record. Checkpoints and W&B caches inside them are not committed to Git.

**Weights & Biases:** runs log to the W&B project **`new_experiments`** (group `baseline-v2`; smoke runs `baseline-v2-smoke`). The old pilot project `temporal-composition-gym` is refused. Authenticate once on each machine with:

```bash
wandb login
```

W&B stores the key in `~/.netrc`. The code never reads, prints or stores an API key, and needs no `.env` file.

---

## Repository Layout

| Path | Contents |
|---|---|
| `temporal_comp_gym/` | The three environments (`envs/<name>/core.py`, `objects.py`, `wrappers.py`) and the scripted oracles |
| `agents/` | Baseline training code: entry points, shared runner, reward machines |
| `run_scripts/baseline/` | Launchers, preflight, plan file, protocol and launch docs, validation reports, `results/` |
| `tests/` | Automated tests (`tests/visual/` holds display demos) |
| `play_human/` | Keyboard-playable versions of the tasks |
| `gym_docs/` | Detailed documentation of the environments, package and tests |
