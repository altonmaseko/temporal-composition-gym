# Reinforcement Learning Agents (protocol v2)

This directory holds the training code for the protocol v2 baselines. All six entry-point scripts are thin wrappers around one shared runner, [`common/runner.py`](common/runner.py). Every run's settings (environment, budget, hyperparameters, seeds, evaluation, W&B destination) come from one place, [`common/baseline_plan.py`](common/baseline_plan.py). The frozen protocol is [`run_scripts/baseline/BASELINE_PROTOCOL.md`](../run_scripts/baseline/BASELINE_PROTOCOL.md).

> **Changed from the pilot.** The pilot scripts (per-script `config` dicts, `--env-id`/`--timesteps` flags, `models/` auto-resume, `wandb.init(sync_tensorboard=True)`, the proposition wrapper `rm_wrapper.py`) are gone. `rm_wrapper.py` now only raises an `ImportError` that points to `agents/reward_machines/`. Old pilot results were produced by that code and are historical, not validated results.

---

## 1. Layout

| Path | Contents |
|---|---|
| `train_dqn.py` | DQN (memoryless) and, with `--task-state`, the DQN-TS control |
| `train_ppo.py` | PPO (memoryless) |
| `train_recurrent_ppo.py` | Recurrent PPO (implicit memory) |
| `drqn/train_drqn.py`, `drqn/drqn_core.py` | DRQN entry point and learner (implicit memory) |
| `qrm/train_qrm.py`, `qrm/qrm_core.py` | QRM (delivery) and QRM-Cue (colour) entry point and learner (explicit memory) |
| `ppo_rm/train_ppo_rm.py` | PPO-RM (explicit memory) |
| `reward_machines/` | Event-driven reward machines and `RewardMachineWrapper` |
| `common/baseline_plan.py` | Run matrix, budgets, evaluation, hyperparameters, smoke mode, config hashing |
| `common/runner.py` | Shared training flow: run directory, metadata, W&B, training, evaluation, checkpoints, resume |
| `common/env_factory.py` | Builds every environment with the frozen main-condition settings |
| `common/seeding.py` | Training and evaluation seed sets, per-run random streams |
| `common/evaluation.py` | Greedy frozen-policy evaluation on fixed episode seeds |
| `common/tracking.py` | Episode records, CSV + W&B logging, W&B setup |
| `common/run_io.py` | Run directories, identity, metadata, status, CSV files |
| `common/replay.py` | `FrameReplay`, the frame-based replay buffer used by DRQN and QRM |
| `common/encoders.py` | NatureCNN (RGB images) and MLP encoders used by DRQN and QRM |

---

## 2. Which Algorithm Runs Where

The baseline matrix has 45 main runs plus 10 control runs (55 in total), with seeds 1–5. Each machine runs one memory family.

| Family | Hazardous Delivery (`delivery`) | Sequential Colour (`colour`) | Deferred Maintenance (`maintenance`) |
|---|---|---|---|
| Memoryless | DQN | DQN | PPO |
| Implicit memory | DRQN | DRQN | Recurrent PPO |
| Explicit memory (reward machine) | QRM | QRM-Cue | PPO-RM |
| Task-state control (memoryless machine) | DQN-TS | DQN-TS | – |

Budgets (environment steps, the same for every algorithm in an environment): delivery 1,000,000; colour 3,000,000; maintenance 2,048,000 (a multiple of PPO's 2,048-step rollout).

All agents in one environment receive the same base environment from `common/env_factory.py`:

| Environment | Base observation every agent gets | Actions |
|---|---|---|
| delivery | 8 values: x, y, 4 neighbour sensors, current square, health (delivered and carried counts hidden) | 4 discrete |
| colour | 84×84×3 RGB frame | 4 discrete |
| maintenance | 28 values, fixed scaling (`MaintenanceScaledObservation`) | 2 continuous |

The environment docs (`gym_docs/03`–`11`) describe these observations in detail.

---

## 3. The Algorithms

### DQN (memoryless) — `train_dqn.py`
Stable-Baselines3 `DQN`. Delivery uses `MlpPolicy` (two hidden layers of 128); colour uses `CnnPolicy` (NatureCNN to 512 features, then one hidden layer of 256). It sees only the current observation.

### PPO (memoryless) — `train_ppo.py`
Stable-Baselines3 `PPO` with `MlpPolicy` (separate policy and value networks, two layers of 128 each), maintenance only.

### DRQN (implicit memory) — `drqn/train_drqn.py`, `drqn/drqn_core.py`
A custom Deep Recurrent Q-Network: encoder → LSTM → Q-values.
- **Delivery:** MLP encoder (128) and LSTM (128).
- **Colour:** NatureCNN on RGB frames (512) and LSTM (256). Frames are kept in colour and only rescaled to [0, 1].
- The recurrent state advances on every step, including random exploration steps.
- Training samples sequences from `FrameReplay` (sized in frames; images stored once as `uint8`). Sequences never cross episode boundaries. Each starts from the stored recurrent state of its first frame, with a burn-in context: 16 burn-in + 32 trained steps for delivery, 32 + 96 for colour. Half of the colour sequences start at the episode start.
- Padding and burn-in steps carry no loss, but gradients do flow through the burn-in frames.
- Episodes ended by the time limit (truncation) bootstrap; only real termination stops bootstrapping.

### Recurrent PPO (implicit memory) — `train_recurrent_ppo.py`
`sb3-contrib` `RecurrentPPO` with `MlpLstmPolicy` (LSTM of 128, separate critic LSTM, one layer of 128 for policy and value), maintenance only.

### QRM (explicit memory, delivery) — `qrm/train_qrm.py --env delivery`, `qrm/qrm_core.py`
Q-learning for Reward Machines. One shared network (MLP encoder 128 → hidden 128) outputs a separate set of Q-values (a "head") for each reward-machine state. The agent acts with the head of the current RM state. Each replayed transition updates every head for which the reward machine says the environment would have behaved the same way (counterfactual updates), in one vectorised pass. Targets stop bootstrapping on the RM's terminal flag, so time-limit truncations still bootstrap.

### QRM-Cue (the explicit Sequential Colour baseline) — `qrm/train_qrm.py --env colour`
The same QRM learner with a NatureCNN encoder (512 → hidden 256) and the cue-recording reward machine (Section 4.3). The same script selects QRM or QRM-Cue from `--env`. It is reported as "QRM-Cue (QRM with a cue-recording RM)" (protocol decision D3).

### PPO-RM (explicit memory, maintenance) — `ppo_rm/train_ppo_rm.py`
Stable-Baselines3 `PPO` with the same network and settings as PPO, but its observation has the maintenance reward machine's 8 task-state features appended (28 + 8 = 36 values). PPO is on-policy, so there are no counterfactual updates.

### DQN-TS (task-state control) — `train_dqn.py --task-state`
Plain DQN given the reward-machine task state as extra input. It is a control: it separates "being given the task memory" from QRM's counterfactual learning (protocol decision D4).
- **Delivery:** the 7-value one-hot RM state is appended to the observation (8 + 7 = 15 values).
- **Colour:** a dictionary observation `{"image": frame, "task_state": 40-value one-hot head}` with `MultiInputPolicy`, so the task state joins after the CNN.

It is trained on the environment's reward.

---

## 4. Reward Machines and Events — `reward_machines/`

Each reward machine reads only the environment's `info["events"]` stream and its fixed configuration (rewards, number of packages, vocabulary size and so on). It never reads the environment's hidden state. **There is no reward shaping:** the RM's reward equals the environment's reward on every step. `RewardMachineWrapper` checks this and raises an error if the RM reward or terminal flag ever differs from the environment's.

`RewardMachineWrapper` modes, as used by the runner:

| Agent | Observation | Reward returned |
|---|---|---|
| QRM, QRM-Cue | unchanged; the RM state and head are in `info["rm"]` | RM reward (equal to the env reward) |
| PPO-RM | RM features appended (`"append"`) | RM reward |
| DQN-TS delivery | RM features appended (`"append"`) | env reward |
| DQN-TS colour | `{"image", "task_state"}` (`"dict"`) | env reward |

### 4.1 Delivery RM (`delivery.py`, used by QRM and DQN-TS)
- **Events used:** `pickup`, `package_contact`, `delivery`, `destination_contact`, `goal_contact`, `lava`, `damage`, `health_depleted`, `heal`.
- **State:** `(d, c)`, deliveries so far and packages carried, with `d + c ≤ 3` and `c ≤ 1`. That is 7 live states (7 Q heads), plus a terminal state.
- **Transitions and rewards:** `pickup` → c+1; `delivery` → d+1, c−1, +10; `goal_contact` → terminal (+50 if d = 3, else −50); `lava` → terminal, −50; `health_depleted` → terminal, −51; `damage` → −1.
- **Counterfactual rule:** a head is updated only if the environment would have produced the same event from that state. A pickup needs spare capacity and a package left; a package contact without pickup needs full hands; a delivery needs a carried package; a destination contact without delivery needs empty hands. All other events are valid for every head.

### 4.2 Maintenance RM (`maintenance.py`, used by PPO-RM)
- **Events used:** `code_collected(k)`, `service_attempt(k)`, `readiness_on`, `stage_exit`, `stage_exit_incomplete`, `final_exit`, `boundary_contact`, `off_path`.
- **State:** the set of held codes, the stage index and the ready bit.
- **Features (8):** held codes as a multi-hot vector (3), stage as a one-hot vector (4), ready (1).

### 4.3 Cue-recording RM (`colour.py`, used by QRM-Cue and DQN-TS colour)
- **Events used:** `cue(colour)`, `delay_start`, `nav_start`, `wall_touch(colour)`, `room_reset`. The first `cue` arrives in the reset `info`.
- **State:** the phase (cue, delay or navigation), the colour IDs recorded from `cue` events in order, and the number of walls completed. It is built from events alone; the environment's hidden sequence is never read.
- **Heads (40 at the defaults):** 5 for the cue phase (by cues seen), 5 for the delay phase (by cues seen), and 6 × 5 for navigation, indexed by the current target colour and the number of walls remaining.
- **Counterfactual rule:**
  - A navigation step with no wall touch updates every navigation head.
  - A wrong wall `w` updates the navigation heads whose target is not `w`.
  - A correct wall updates heads with that target and at least 2 walls left.
  - Completing the sequence updates only the head with that target and 1 wall left.
  - Cue and delay steps update only the actual head.
- **Requirements:** the head abstraction assumes randomised wall colours, a vocabulary of at least 4 and finite episodes. `CueRecordingRM.from_env` raises a `ValueError` otherwise.

### Task memory each explicit agent is given beyond the normal observation

| Agent | Given | Still has to work out itself |
|---|---|---|
| QRM (delivery) | Deliveries so far and whether a package is carried (both hidden in the observation), as the active head; ground-truth event labels (pickup vs delivery). | Where it has seen packages, destinations and hazards. |
| QRM-Cue (colour) | The cue sequence as symbolic colour IDs, the cue boundaries (one event per item, including repeats), and its progress through the sequence. In navigation, the head encodes the current target colour and walls remaining. | Which wall shows the target colour in each room, and how to reach it. |
| PPO-RM (maintenance) | Held code types, stage index and ready bit, appended to the observation. Readiness is also visible in the base observation. | Where stations and codes are in each stage, and how to fly there. |
| DQN-TS (control) | The same RM state as QRM / QRM-Cue, as input features, without counterfactual updates. | As for QRM / QRM-Cue. |

---

## 5. How a Trainer Is Configured

Nothing is configured by editing the scripts. `common/baseline_plan.py` resolves a complete configuration for each `(environment, algorithm, seed, mode)`:

- **Environment:** the frozen main-condition settings from `common/env_factory.py`.
- **Hyperparameters:** the untuned common defaults from protocol section 6.2.
  - DQN family: γ 0.99, buffer 100,000, train every 4 steps, batch 32, ε from 1.0 to 0.05 over the first 20% of the budget, gradient clip 10. Learning rate 5e-4 (delivery) or 1e-4 (colour). Learning starts at 10,000 (delivery) or 20,000 (colour) steps. Target update every 2,000 (delivery) or 10,000 (colour) steps.
  - PPO family: n_steps 2,048, batch 64, 10 epochs, learning rate 3e-4, γ 0.99, GAE λ 0.95, clip 0.2, entropy 0.0.
  - DRQN and QRM use their own config classes, `DRQN_DEFAULTS` and `QRM_DEFAULTS`. They match the DQN values above for learning rate, warm-up, target-update interval, ε schedule, γ, train frequency and gradient clip, with a 100,000-frame replay. Batch sizes differ: DRQN samples 8 sequences per update (colour: 4), QRM 32 transitions.
- **Evaluation:** periodic evaluation at step 0 and every 5% of the budget (21 points) on 50 episodes (maintenance: 20), and a final evaluation on 200 episodes (maintenance: 100). Both use fixed held-out episode seeds (1,000,000+ and 2,000,000+), the same for every algorithm, and a greedy policy whose recurrent state resets each episode.
- **Checkpoints:** a policy snapshot and a restart checkpoint every 10% of the budget. SB3 checkpoints are taken at rollout boundaries.
- **Seeds:** the training seed must be 1–10 (1–5 for the baseline; 6–10 are reserved for expansion). All random streams (torch, NumPy, replay, exploration, environment) derive from it.
- **Config hash:** a SHA-256 hash of the resolved configuration (excluding the W&B destination) identifies the run.

`--set KEY.PATH=VALUE` may override an existing entry of the resolved configuration (for example `--set algo.hyperparams.learning_rate=1e-4`). Overrides are recorded and change the config hash, so the manifest reports such runs as `nonstandard`.

Smoke mode (`--mode smoke`) uses tiny budgets (delivery 4,000, colour 3,000, maintenance 4,096 steps), 2 evaluation points, shortened schedules and a 600-step maintenance limit. It is for pipeline checks only. Its outputs go under `results/v2/smoke/` and to the W&B group `baseline-v2-smoke`, apart from production.

The full resolved plan (every run's configuration and hash) is written to [`run_scripts/baseline/baseline_plan.json`](../run_scripts/baseline/baseline_plan.json):

```bash
D:\miniconda3\envs\temporal_gym\python.exe -m agents.common.baseline_plan write
D:\miniconda3\envs\temporal_gym\python.exe -m agents.common.baseline_plan list --role explicit
```

---

## 6. How to Launch

Production runs are started by the three machine launchers, not by calling the scripts one at a time. See [`run_scripts/baseline/README.md`](../run_scripts/baseline/README.md) and [`run_scripts/baseline/PRODUCTION_LAUNCH_v2.md`](../run_scripts/baseline/PRODUCTION_LAUNCH_v2.md) for setup, preflight, the per-machine commands, restart and collection.

A single run can be started by hand from any working directory. `--env` takes `delivery`, `colour` or `maintenance` (only those the algorithm supports), and `--seed` is required:

```bash
python agents/train_dqn.py --env delivery --seed 1
python agents/train_dqn.py --env colour --seed 1 --task-state          # DQN-TS
python agents/train_ppo.py --env maintenance --seed 1
python agents/train_recurrent_ppo.py --env maintenance --seed 1
python agents/drqn/train_drqn.py --env colour --seed 1 --device cuda --require-gpu
python agents/qrm/train_qrm.py --env delivery --seed 1                 # QRM
python agents/qrm/train_qrm.py --env colour --seed 1                   # QRM-Cue
python agents/ppo_rm/train_ppo_rm.py --env maintenance --seed 1
```

(On the development PC, use `D:\miniconda3\envs\temporal_gym\python.exe` in place of `python`.)

Common options:

| Option | Meaning |
|---|---|
| `--mode full` / `smoke` | Full run (default) or labelled smoke check |
| `--print-config` | Print the resolved configuration and its hash, then exit |
| `--resume RUN_DIR` | Continue that run directory (an episode-boundary restart; refused if the config hash, seed, env or algorithm differ, the run is already complete, or there is no restart checkpoint) |
| `--device auto` / `cpu` / `cuda` / `cuda:N`, `--require-gpu` | Device choice; `--require-gpu` fails rather than running on the CPU |
| `--threads N` | Torch CPU threads |
| `--wandb-mode online` / `offline` / `disabled` | Default: online for full runs, offline for smoke runs |
| `--wandb-project`, `--wandb-entity` | W&B destination (default project `new_experiments`; the pilot project `temporal-composition-gym` is refused) |
| `--results-root DIR` | Where run directories go (default `run_scripts/baseline/results`) |
| `--set KEY.PATH=VALUE` | Override an existing config entry (recorded; changes the hash) |

The launchers use the CPU for delivery and maintenance runs and require CUDA for colour runs. Exit codes: 0 complete, 1 failed, 130 interrupted.

---

## 7. Where Outputs Go

Each run gets a new directory that is never reused:

```
run_scripts/baseline/results/v2/<full|smoke>/<env>/<algo>/seed<s>__<cfghash10>__<utc>-<rand>/
    config.resolved.json  identity.json  metadata.json  status.json  final_summary.json
    train_episodes.csv  train_updates.csv  resources.csv
    eval_periodic.csv  eval_points.csv  eval_final.csv
    checkpoints/policy_step<N>.{zip,pt}   checkpoints/restart/   logs/   wandb/
```

- **Official numbers** are the local CSV and JSON files. W&B receives the same numbers but is not the record.
- `status.json` has `state` set to `running`, `complete`, `failed` or `interrupted`. Only runs that are `complete` and `verified` count. A run is marked complete only after the budget, the number of evaluations and the final-evaluation episode count have been checked.
- `metadata.json` records the git revision, dirty files and diff hash, package versions, hardware, the selected device and the command line. No secrets are recorded.
- **W&B:**
  - Project `new_experiments`, group `baseline-v2` (smoke: `baseline-v2-smoke`).
  - Job type `<env>/<algo>`, run name equal to the run ID, and `global_step` (environment steps) as the x-axis.
  - Authentication is only through `wandb login` on each machine (stored by W&B in `~/.netrc`). The code never reads, prints or stores an API key.
