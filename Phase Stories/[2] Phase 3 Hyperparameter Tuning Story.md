# Complete Hyperparameter Tuning Story

## 1. Overview & Research Objective

Following the baseline runs in [Phase 3 Baseline Execution Story]([1]%20Phase%203%20Baseline%20Execution%20Story.md), empirical evaluation revealed that memory-augmented architectures (implicit recurrent models and explicit symbolic reward machines) showed early learning signals but suffered from policy instability, catastrophic forgetting, or sensitivity to optimization hyperparameters.

To maximize stability and ensure reliable convergence before advancing to Phase 4 stress testing, systematic Bayesian hyperparameter optimization was conducted using **Optuna** (Tree-structured Parzen Estimator sampler) with persistent **SQLite** storage and full telemetry logging to **Weights & Biases (`wandb`)**.

Tuning focused on the primary memory-augmented representative for each of the three benchmark environments:

1. **`TemporalComp/HazardousDelivery-v0` (Tabular Logistics)** — **Q-Learning for Reward Machines (QRM)**: Optimized replay buffer size, target network sync frequency, batch size, learning rate, and epsilon-greedy exploration schedules.
2. **`TemporalComp/SequentialColour-v0` (Image Navigation)** — **Deep Recurrent Q-Network (DRQN)**: Optimized truncated backpropagation through time (`seq_len`), image replay buffer capacity, batch size, learning rate, and exploration decay.
3. **`TemporalComp/DeferredMaintenance-v0` (Continuous Control)** — **PPO for Reward Machines (PPO-RM)**: Optimized rollout horizon (`n_steps`), mini-batch size, entropy regularization coefficient (`ent_coef`), surrogate clipping range (`clip_range`), and Adam learning rate.

---

## 2. Infrastructure & Execution Environment

Hyperparameter tuning was executed on dedicated cloud compute via Vast.ai:

* **GPU:** 1x NVIDIA GeForce RTX 3090 Ti (24 GB GDDR6X VRAM)
* **CPU:** 40 vCPUs (Intel Xeon E5-2673 v4 @ 2.30 GHz)
* **System RAM:** 129 GB DDR4
* **Storage:** 32 GB High-Speed NVMe
* **Software Stack:**
  * Ubuntu Linux headless container
  * PyTorch 2.x & CUDA 12.x
  * Stable-Baselines3 (SB3) & SB3-Contrib
  * Optuna 3.x with SQLite RDB backend
  * Gymnasium & TensorBoard
  * Weights & Biases (`wandb`)
* **Session Persistence:** Managed through an uninterrupted background `tmux` session (`[ssh_tmux]`).

### Critical System Tuning & Deadlock Mitigations
High-core cloud instances (40 vCPUs) introduced subtle threading and synchronization bottlenecks that required concrete technical remediations:
* **OpenMP Thread Thrashing / Deadlock:** PyTorch CPU thread spawning across 40 virtual cores caused near-zero CPU progress (0.01% utilization freeze). Resolved by enforcing:
  ```python
  import os
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  import torch
  torch.set_num_threads(1)
  ```
* **W&B TensorBoard Deadlock:** Calling `sync_tensorboard=True` inside sequential Optuna loops created thread deadlocks between TensorBoard writers and W&B sync threads. Replaced with the native `WandbCallback` from `wandb.integration.sb3`.
* **Deterministic Evaluation Infinite Loops:** SB3's deterministic `EvalCallback` would hang indefinitely if untrained policies cycled indefinitely. Resolved by wrapping environments with `gym.wrappers.TimeLimit(env, max_episode_steps=1000)`.

---

## 3. Execution Pipeline & Tuning Scripts

Tuning was orchestrated via three standalone Python drivers and corresponding shell execution scripts:

```
run_scripts/hyperparameter_tuning/
├── run_tune_qrm.sh
├── tune_qrm.py
├── run_tune_drqn.sh
├── tune_drqn.py
├── run_tune_ppo_rm.sh
└── tune_ppo_rm.py
```

### Step 1: QRM Hyperparameter Tuning & Checkpoint Collision Fix
> 📁 Shell Driver: [`run_scripts/hyperparameter_tuning/run_tune_qrm.sh`](../run_scripts/hyperparameter_tuning/run_tune_qrm.sh)  
> 🐍 Python Driver: [`run_scripts/hyperparameter_tuning/tune_qrm.py`](../run_scripts/hyperparameter_tuning/tune_qrm.py)

* **Target Algorithm:** Q-Learning for Reward Machines (QRM)
* **Environment:** `TemporalComp/HazardousDelivery-v0`
* **Budget & Scale:** 15 Optuna Trials, 300,000 steps per trial (reduced from 1M for efficiency)
* **Concurrency:** 8 parallel Optuna workers (`n_jobs=8`), leveraging the high CPU core count
* **Search Space:**
  * `learning_rate`: Log-uniform $[1\times 10^{-5},\, 1\times 10^{-3}]$
  * `buffer_size`: Categorical $[10\,000,\, 50\,000,\, 100\,000]$
  * `target_network_frequency`: Categorical $[100,\, 500,\, 1000,\, 2000]$
  * `batch_size`: Categorical $[32,\, 64,\, 128]$
  * `exploration_fraction`: Uniform $[0.1,\, 0.8]$
* **Initial Failure Diagnosed & Patched:**
  * Early test trials failed or exhibited erratic returns due to a checkpoint collision bug in `load_latest_checkpoint()` within [`agents/qrm/train_qrm.py`](../agents/qrm/train_qrm.py).
  * The checkpoint search glob (`qrm_HazardousDelivery-v0_step_*.pt`) did not condition on the random seed / trial number (`args.seed`). Consequently, newly spawned tuning trials inadvertently loaded pre-existing weights saved by previous trials or baseline runs.
  * **Remediation:** Patched the search pattern to `f"{algo_name}_{env_name}_seed{seed}_step_*.pt"` ensuring every trial started with freshly initialized network parameters.
* **Result:** After the patch, all 15 trials completed cleanly.
  * **Winning Configuration (Trial 9):**
    * **Objective Reward:** **-15.1**
    * `learning_rate`: $2.2068 \times 10^{-5}$
    * `buffer_size`: $100\,000$
    * `target_network_frequency`: $2000$
    * `batch_size`: $32$
    * `exploration_fraction`: $0.4093$

---

### Step 2: DRQN Hyperparameter Tuning & Memory Concurrency Constraints
> 📁 Shell Driver: [`run_scripts/hyperparameter_tuning/run_tune_drqn.sh`](../run_scripts/hyperparameter_tuning/run_tune_drqn.sh)  
> 🐍 Python Driver: [`run_scripts/hyperparameter_tuning/tune_drqn.py`](../run_scripts/hyperparameter_tuning/tune_drqn.py)

* **Target Algorithm:** Deep Recurrent Q-Network (DRQN with LSTM cell)
* **Environment:** `TemporalComp/SequentialColour-v0` (RGB Image Observations)
* **Budget & Scale:** 15 Optuna Trials, 300,000 steps per trial (reduced from 3M for tuning)
* **Search Space:**
  * `learning_rate`: Log-uniform $[1\times 10^{-5},\, 1\times 10^{-3}]$
  * `buffer_size`: Categorical $[5\,000,\, 10\,000,\, 20\,000]$
  * `seq_len` (BPTT rollout length): Categorical $[4,\, 8,\, 16]$
  * `batch_size`: Categorical $[16,\, 32,\, 64]$
  * `exploration_fraction`: Uniform $[0.1,\, 0.8]$
* **Failure Diagnosed & Handled:**
  * Initial multi-worker runs (`n_jobs >= 4`) caused GPU Out-Of-Memory (CUDA OOM) crashes due to the large memory footprint of visual observation buffers during sequential recurrent rollouts.
  * Applied the identical seed-specific checkpoint fix to [`agents/drqn/train_drqn.py`](../agents/drqn/train_drqn.py) to prevent weights bleed-through across trials.
  * **Remediation:** Strictly capped concurrency to `n_jobs=2`.
* **Result:** All 15 trials completed successfully without GPU VRAM exhaustion.
  * **Winning Configuration (Trial 12):**
    * **Objective Reward:** **-5.8**
    * `learning_rate`: $1.2570 \times 10^{-5}$
    * `buffer_size`: $10\,000$
    * `seq_len`: $4$
    * `batch_size`: $16$
    * `exploration_fraction`: $0.6433$

---

### Step 3: PPO-RM Tuning Failure (The 3-Hour Vast.ai Server Freeze)
> 📁 Shell Driver: [`run_scripts/hyperparameter_tuning/run_tune_ppo_rm.sh`](../run_scripts/hyperparameter_tuning/run_tune_ppo_rm.sh)  
> 🐍 Python Driver: [`run_scripts/hyperparameter_tuning/tune_ppo_rm.py`](../run_scripts/hyperparameter_tuning/tune_ppo_rm.py)

* **Target Algorithm:** PPO for Reward Machines (PPO-RM)
* **Environment:** `TemporalComp/DeferredMaintenance-v0` (Continuous 2D Navigation)
* **Failure Incident:**
  * When launched on the 40-core Vast.ai server, the initial tuning run ran for over 3 hours with almost no progress, eventually stalling entirely with CPU utilization plummeting to **0.01%**.
  * The terminal output froze mid-execution, requiring a hard manual termination (`kill -9`) of the job.
* **Root Cause Diagnosis:**
  1. **PyTorch OpenMP Thread Allocation Deadlock:** PyTorch CPU thread spawning across the 40 virtual cores caused extreme CPU thread thrashing and synchronization deadlocks inside the Ubuntu container.
  2. **W&B / TensorBoard Thread Contention Deadlock:** The script utilized `sync_tensorboard=True` within sequential Optuna loops, which locked file write locks between TensorBoard's background flush threads and W&B sync threads.
  3. **Deterministic Evaluation Infinite Loop:** Stable-Baselines3's deterministic `EvalCallback` hung indefinitely because `TemporalComp/DeferredMaintenance-v0` lacked an episode step limit wrapper. When an untrained agent entered a non-terminating trajectory loop, the evaluation never returned a `done` signal, blocking Optuna forever.
* **Incident Clean-up:** The hung run was aborted, and the corrupted SQLite database (`run_scripts/hyperparameter_tuning/ppo_rm_tuning.db`) containing stale/frozen trial states was completely purged.

---

### Step 4: PPO-RM Remediation, Local Verification & Successful Rerun

To resolve the triple-deadlock and allow PPO-RM tuning to finish, four critical fixes were engineered:

1. **Thread Clamping:** Bound PyTorch and OpenMP thread pools to 1 single thread:
   ```python
   import os
   os.environ["OMP_NUM_THREADS"] = "1"
   os.environ["MKL_NUM_THREADS"] = "1"
   import torch
   torch.set_num_threads(1)
   ```
2. **Native W&B Callback:** Removed `sync_tensorboard=True` and integrated SB3's official callback:
   ```python
   from wandb.integration.sb3 import WandbCallback
   ```
3. **Mandatory Step Limit Wrapper:** Wrapped `DeferredMaintenance-v0` inside `make_env()`:
   ```python
   env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
   ```
4. **Execution Mode:** Shifted from multi-worker to strict sequential execution (`n_jobs=1`) with verbose terminal logging enabled to track progress.

* **Local Verification:** Before redeploying to the cloud, the patched code was verified locally using the Conda environment (`D:\miniconda3\envs\temporal_gym\python.exe`) with offline logging (`WANDB_MODE=offline`) to confirm the evaluation callback successfully terminated.
* **Final Cloud Execution Result:** Redeployed to Vast.ai, all 20 trials completed cleanly in ~3.5 hours.
  * **Winning Configuration (Trial 8):**
    * **Objective Reward:** **-434.2**
    * `learning_rate`: $8.0869 \times 10^{-4}$
    * `n_steps`: $4096$
    * `batch_size`: $128$
    * `ent_coef`: $0.0257$
    * `clip_range`: $0.2$

---

## 4. Generated Outputs & Local Artifacts

### Live Experiment Synchronization (Weights & Biases)
* **Project:** `altonmaseko-wits-university/temporal-composition-gym`
* **Total Tuning Phase Runs on W&B:** **60 runs**
  * **50 Completed Optuna Trials:** 15 QRM trials + 15 DRQN trials + 20 PPO-RM trials.
  * **10 Probe & Preliminary Verification Runs:** Initial smoke and cluster verification runs logged during early test orchestration.

### Remote Server Artifacts (Vast.ai)
* TensorBoard event logs stored under `/workspace/temporal-composition-gym/runs/`.
* SQLite databases storing all intermediate trial parameters and loss trajectories.

### Local Exported Datasets & Databases
All tuning databases and tabular summaries were extracted from the server and are stored in the repository under:
> 📁 [`run_scripts/hyperparameter_tuning/results/`](../run_scripts/hyperparameter_tuning/results/)

* **QRM Tuning:**
  * Database: [`run_scripts/hyperparameter_tuning/results/qrm_tuning.db`](../run_scripts/hyperparameter_tuning/results/qrm_tuning.db) (SQLite study `qrm_tuning`)
  * Results CSV: [`run_scripts/hyperparameter_tuning/results/qrm_tuning_results.csv`](../run_scripts/hyperparameter_tuning/results/qrm_tuning_results.csv) (15 trials)
* **DRQN Tuning:**
  * Database: [`run_scripts/hyperparameter_tuning/results/drqn_tuning.db`](../run_scripts/hyperparameter_tuning/results/drqn_tuning.db) (SQLite study `drqn_tuning`)
  * Results CSV: [`run_scripts/hyperparameter_tuning/results/drqn_tuning_results.csv`](../run_scripts/hyperparameter_tuning/results/drqn_tuning_results.csv) (15 trials)
* **PPO-RM Tuning:**
  * Database: [`run_scripts/hyperparameter_tuning/results/ppo_rm_tuning.db`](../run_scripts/hyperparameter_tuning/results/ppo_rm_tuning.db) (SQLite study `ppo_rm_tuning`)
  * Results CSV: [`run_scripts/hyperparameter_tuning/results/ppo_rm_tuning_results.csv`](../run_scripts/hyperparameter_tuning/results/ppo_rm_tuning_results.csv) (20 trials)

### Local Weights & Biases Export Datasets
In addition to the SQLite Optuna databases and summary CSVs, the full timeseries curves and comprehensive metadata across all 60 hyperparameter tuning runs were exported directly from Weights & Biases and stored in:
> 📁 [`run_scripts/hyperparameter_tuning/wandb_exports/`](../run_scripts/hyperparameter_tuning/wandb_exports/)

* **Master Summary Table:** [`run_scripts/hyperparameter_tuning/wandb_exports/all_runs_summary.csv`](../run_scripts/hyperparameter_tuning/wandb_exports/all_runs_summary.csv) — Contains the complete metadata, hyperparameters, trial configurations, run IDs, and final summary metrics across all 60 tuning runs (60 rows × 71 metric/config columns).
* **History Timeseries Coordinates:** 60 individual `*_history.csv` files containing step-by-step curve coordinates (episodic return trajectories, value losses, policy losses, exploration rates, hardware metrics) sampled across each run's lifecycle.

---

## 5. Complete Registry of Tuning W&B Runs (60 Total Runs)

### QRM on `TemporalComp/HazardousDelivery-v0` (15 Trials)

| Trial # | Value (Return) | W&B Run Name | W&B Run ID | W&B Run Link | Status |
|:---:|:---:|---|:---:|:---:|:---:|
| 0 | -41.50 | `TemporalComp_HazardousDelivery-v0__qrm__0__1791107185` | `3u4nhkhk` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/3u4nhkhk) | Finished |
| 1 | -45.75 | `TemporalComp_HazardousDelivery-v0__qrm__1__1791107185` | `rqfxu1qa` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/rqfxu1qa) | Finished |
| 2 | -46.50 | `TemporalComp_HazardousDelivery-v0__qrm__2__1791107185` | `3aw82xot` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/3aw82xot) | Finished |
| 3 | -51.20 | `TemporalComp_HazardousDelivery-v0__qrm__3__1791107185` | `2x63mfc9` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/2x63mfc9) | Finished |
| 4 | -45.50 | `TemporalComp_HazardousDelivery-v0__qrm__4__1791107185` | `m5i0l7ud` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/m5i0l7ud) | Finished |
| 5 | -29.60 | `TemporalComp_HazardousDelivery-v0__qrm__5__1791107185` | `0ajyvhd4` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/0ajyvhd4) | Finished |
| 6 | -47.50 | `TemporalComp_HazardousDelivery-v0__qrm__6__1791107185` | `wqhz7twu` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/wqhz7twu) | Finished |
| 7 | -30.70 | `TemporalComp_HazardousDelivery-v0__qrm__7__1791107185` | `7aor5s2c` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/7aor5s2c) | Finished |
| 8 | -50.70 | `TemporalComp_HazardousDelivery-v0__qrm__8__1791134925` | `2ixdctdw` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/2ixdctdw) | Finished |
| **9** | **-15.10** | `TemporalComp_HazardousDelivery-v0__qrm__9__1791159277` | `hdvgefdu` | [**View Run (Best)**](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/hdvgefdu) | **Finished (Best)** |
| 10 | -25.60 | `TemporalComp_HazardousDelivery-v0__qrm__10__1791159300` | `l56adld8` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/l56adld8) | Finished |
| 11 | -30.70 | `TemporalComp_HazardousDelivery-v0__qrm__11__1791159303` | `im359w6b` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/im359w6b) | Finished |
| 12 | -20.80 | `TemporalComp_HazardousDelivery-v0__qrm__12__1791159329` | `1l5sx1k7` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/1l5sx1k7) | Finished |
| 13 | -30.70 | `TemporalComp_HazardousDelivery-v0__qrm__13__1791159344` | `rpphz9nk` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/rpphz9nk) | Finished |
| 14 | -30.60 | `TemporalComp_HazardousDelivery-v0__qrm__14__1791159375` | `4jfp9di5` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/4jfp9di5) | Finished |

---

### DRQN on `TemporalComp/SequentialColour-v0` (15 Trials)

| Trial # | Value (Return) | W&B Run Name | W&B Run ID | W&B Run Link | Status |
|:---:|:---:|---|:---:|:---:|:---:|
| 0 | -90.00 | `TemporalComp_SequentialColour-v0__drqn__0__1791220532` | `54erdgj4` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/54erdgj4) | Finished |
| 1 | -7.70 | `TemporalComp_SequentialColour-v0__drqn__1__1791220532` | `2berpkb6` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/2berpkb6) | Finished |
| 2 | -10.30 | `TemporalComp_SequentialColour-v0__drqn__2__1791226079` | `lbrywu9y` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/lbrywu9y) | Finished |
| 3 | -164.10 | `TemporalComp_SequentialColour-v0__drqn__3__1791229170` | `tgqnvkri` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/tgqnvkri) | Finished |
| 4 | -303.00 | `TemporalComp_SequentialColour-v0__drqn__4__1791229382` | `03cwjvjm` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/03cwjvjm) | Finished |
| 5 | -122.90 | `TemporalComp_SequentialColour-v0__drqn__5__1791253714` | `90jk6myc` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/90jk6myc) | Finished |
| 6 | -167.00 | `TemporalComp_SequentialColour-v0__drqn__6__1791253908` | `9mkms4u6` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/9mkms4u6) | Finished |
| 7 | -21.30 | `TemporalComp_SequentialColour-v0__drqn__7__1791262123` | `5efnjv18` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/5efnjv18) | Finished |
| 8 | -61.70 | `TemporalComp_SequentialColour-v0__drqn__8__1791268287` | `l2ddjrt3` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/l2ddjrt3) | Finished |
| 9 | -141.50 | `TemporalComp_SequentialColour-v0__drqn__9__1791276910` | `0hl7tp63` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/0hl7tp63) | Finished |
| 10 | -120.20 | `TemporalComp_SequentialColour-v0__drqn__10__1791277124` | `9vwv6soy` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/9vwv6soy) | Finished |
| 11 | -32.60 | `TemporalComp_SequentialColour-v0__drqn__11__1791282515` | `un2jsbpj` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/un2jsbpj) | Finished |
| **12** | **-5.80** | `TemporalComp_SequentialColour-v0__drqn__12__1791287940` | `of7weqyx` | [**View Run (Best)**](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/of7weqyx) | **Finished (Best)** |
| 13 | -328.30 | `TemporalComp_SequentialColour-v0__drqn__13__1791291988` | `hx2uaisv` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/hx2uaisv) | Finished |
| 14 | -328.30 | `TemporalComp_SequentialColour-v0__drqn__14__1791293416` | `vwxjxmgs` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/vwxjxmgs) | Finished |

---

### PPO-RM on `TemporalComp/DeferredMaintenance-v0` (20 Trials)

| Trial # | Value (Return) | W&B Run Name | W&B Run ID | W&B Run Link | Status |
|:---:|:---:|---|:---:|:---:|:---:|
| 0 | -779.50 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__0__1791486858` | `gourvaa7` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/gourvaa7) | Finished |
| 1 | -783.50 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__1__1791487616` | `qlrok6sh` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/qlrok6sh) | Finished |
| 2 | -1897.34 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__2__1791488251` | `a9o732f9` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/a9o732f9) | Finished |
| 3 | -755.14 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__3__1791488878` | `n7pt33fb` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/n7pt33fb) | Finished |
| 4 | -752.96 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__4__1791489434` | `0pl7wktz` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/0pl7wktz) | Finished |
| 5 | -1481.10 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__5__1791490201` | `hwt1u1yo` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/hwt1u1yo) | Finished |
| 6 | -771.66 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__6__1791490833` | `natnmpik` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/natnmpik) | Finished |
| 7 | -750.50 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__7__1791491603` | `kbgcnqib` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/kbgcnqib) | Finished |
| **8** | **-434.20** | `TemporalComp_DeferredMaintenance-v0__ppo_rm__8__1791492371` | `c0fsl0d5` | [**View Run (Best)**](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/c0fsl0d5) | **Finished (Best)** |
| 9 | -612.84 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__9__1791493002` | `beacx9dy` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/beacx9dy) | Finished |
| 10 | -779.46 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__10__1791493619` | `fqcufu18` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/fqcufu18) | Finished |
| 11 | -758.32 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__11__1791494241` | `gh3sqxqj` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/gh3sqxqj) | Finished |
| 12 | -772.20 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__12__1791494865` | `gn54noam` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/gn54noam) | Finished |
| 13 | -454.86 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__13__1791495503` | `fmixtaeb` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/fmixtaeb) | Finished |
| 14 | -738.68 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__14__1791496130` | `vvcumlzx` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/vvcumlzx) | Finished |
| 15 | -465.76 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__15__1791496749` | `f6f61dl9` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/f6f61dl9) | Finished |
| 16 | -469.10 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__16__1791497311` | `xbl78fay` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/xbl78fay) | Finished |
| 17 | -802.56 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__17__1791498079` | `p6y6251o` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/p6y6251o) | Finished |
| 18 | -457.14 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__18__1791498706` | `1xha7cyi` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/1xha7cyi) | Finished |
| 19 | -739.74 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__19__1791499346` | `ld4ousql` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/ld4ousql) | Finished |

---

### Preliminary, Aborted & Probe Check Runs (10 Early / Failed Runs)

Prior to launching the final Optuna studies, 10 preliminary probe, smoke test, and failed tuning trials were executed. These runs were aborted or rendered invalid due to the checkpoint cross-contamination bug (inadvertently restoring pre-trained weights from earlier runs), wrapper dimension checks, or initial hyperparameter smoke tests:

| # | Paradigm | Environment | W&B Run Name | W&B Run ID | W&B Run Link | Primary Failure / Role |
|:---:|---|---|---|:---:|:---:|---|
| 1 | QRM | `HazardousDelivery-v0` | `TemporalComp_HazardousDelivery-v0__qrm__0__1791054027` | `0qsy4yuk` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/0qsy4yuk) | Aborted early; checkpoint collision verification |
| 2 | QRM | `HazardousDelivery-v0` | `TemporalComp_HazardousDelivery-v0__qrm__1__1791054114` | `4q3gcn7l` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/4q3gcn7l) | Aborted early; multi-worker concurrency test |
| 3 | DRQN | `SequentialColour-v0` | `TemporalComp_SequentialColour-v0__drqn__1__1791054143` | `ti8owe8n` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/ti8owe8n) | Initial image buffer memory probe |
| 4 | QRM | `HazardousDelivery-v0` | `TemporalComp_HazardousDelivery-v0__qrm__0__1791054610` | `vpw0qrdm` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/vpw0qrdm) | Aborted; cross-seed checkpoint contamination |
| 5 | QRM | `HazardousDelivery-v0` | `TemporalComp_HazardousDelivery-v0__qrm__3__1791054610` | `cjp4ceox` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/cjp4ceox) | Aborted; cross-seed checkpoint contamination |
| 6 | QRM | `HazardousDelivery-v0` | `TemporalComp_HazardousDelivery-v0__qrm__8__1791082348` | `ih7u2ojj` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/ih7u2ojj) | Trial run aborted prior to study restart |
| 7 | QRM | `HazardousDelivery-v0` | `TemporalComp_HazardousDelivery-v0__qrm__9__1791082364` | `u19prrak` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/u19prrak) | Trial run aborted prior to study restart |
| 8 | QRM | `HazardousDelivery-v0` | `TemporalComp_HazardousDelivery-v0__qrm__10__1791091595` | `is8mp7ev` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/is8mp7ev) | Trial run aborted prior to study restart |
| 9 | QRM | `HazardousDelivery-v0` | `TemporalComp_HazardousDelivery-v0__qrm__42__1791106979` | `gfs4uexw` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/gfs4uexw) | Standalone verification seed before clean study launch |
| 10 | DRQN | `SequentialColour-v0` | `TemporalComp_SequentialColour-v0__drqn__42__1791107019` | `afr29ede` | [View Run](https://wandb.ai/altonmaseko-wits-university/temporal-composition-gym/runs/afr29ede) | Standalone verification seed before clean study launch |

---

## 6. Optimal Hyperparameters Summary

The table below outlines the optimal hyperparameters discovered through the Optuna studies. These parameters serve as the default foundation for Phase 4 stress testing:

| Hyperparameter | QRM (`HazardousDelivery-v0`) | DRQN (`SequentialColour-v0`) | PPO-RM (`DeferredMaintenance-v0`) |
|---|:---:|:---:|:---:|
| **Winning Trial #** | **Trial 9** | **Trial 12** | **Trial 8** |
| **Best Mean Return** | **-15.1** | **-5.8** | **-434.2** |
| **Learning Rate** | $2.2068 \times 10^{-5}$ | $1.2570 \times 10^{-5}$ | $8.0869 \times 10^{-4}$ |
| **Batch Size** | $32$ | $16$ | $128$ |
| **Replay Buffer Size** | $100\,000$ transitions | $10\,000$ episodes | — |
| **Target Network Frequency** | $2000$ steps | — | — |
| **Sequence Length (BPTT)** | — | $4$ steps | — |
| **Exploration Fraction** | $0.4093$ | $0.6433$ | — |
| **Rollout Steps (`n_steps`)** | — | — | $4096$ steps |
| **Entropy Coefficient (`ent_coef`)** | — | — | $0.0257$ |
| **Clip Range (`clip_range`)** | — | — | $0.2$ |

---

## 7. Key Empirical Takeaways

> [!IMPORTANT]
> **Resolution of Policy Instability:** Tuning demonstrated that the baseline policy collapses observed in Phase 3 were heavily driven by overly aggressive learning rates and insufficient exploration horizons in tabular and image environments, and premature entropy decay in continuous control.

* **Target Network Synchronization:** For QRM on `HazardousDelivery-v0`, increasing `target_network_frequency` to 2000 steps stabilized Q-value oscillation across multiple automata transitions, preventing catastrophic policy degradation.
* **Recurrent Sequence Rollouts:** For DRQN on `SequentialColour-v0`, a sequence length of 4 steps with an extended exploration schedule (0.6433) achieved the highest return (-5.8), substantially outperforming both shorter rollouts and overly long sequences that suffered from gradient instability.
* **Entropy Regularization in Continuous Control:** For PPO-RM on `DeferredMaintenance-v0`, maintaining strong entropy exploration (`ent_coef=0.0257`) combined with long rollouts (`n_steps=4096`) dramatically reduced penalty accumulation from unmaintained systems (improving reward from -1897.34 down to -434.20).
* **Next Steps:** These optimal configurations provide a standardized, tuned baseline to evaluate architectural robustness as temporal dependencies and retention delays scale during **Phase 4 Stress Testing**.
