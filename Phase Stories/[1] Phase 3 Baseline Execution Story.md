# Complete Baseline Execution Story

## 1. Overview & Research Objective
Phase 3 of the **Temporal Composition Gym** benchmark established empirical baselines across three custom environments designed to rigorously test non-Markovian temporal compositions:

1. **`TemporalComp/HazardousDelivery-v0`** — Discrete/tabular logistics task requiring sequential item deliveries while navigating lethal and damage hazard tiles.
2. **`TemporalComp/SequentialColour-v0`** — Image-based visual navigation task requiring agents to inspect and remember target color patterns over extended step horizons.
3. **`TemporalComp/DeferredMaintenance-v0`** — Continuous control navigation task requiring scheduled maintenance interventions to avoid catastrophic penalty accumulation.

The baseline evaluation benchmarks three distinct architectural paradigms across all three environments with 3 independent seeds:
* **Memoryless Baselines:** Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO).
* **Implicit Memory Baselines:** Deep Recurrent Q-Networks (DRQN with LSTM) and Recurrent PPO (PPO with LSTM).
* **Explicit Symbolic Memory Baselines:** Q-Learning for Reward Machines (QRM) and PPO for Reward Machines (PPO-RM).

---

## 2. Infrastructure & Execution Environment

The baseline training jobs were hosted on a dedicated cloud compute instance via Vast.ai:

* **GPU:** 1x NVIDIA GeForce RTX 3090 Ti (24 GB GDDR6X VRAM)
* **CPU:** 40 vCPUs (Intel Xeon E5-2673 v4 @ 2.30 GHz)
* **System RAM:** 129 GB DDR4
* **Storage:** 32 GB High-Speed NVMe
* **Software Stack:**
  * Ubuntu Linux container with PyTorch 2.x & CUDA 12.x
  * Stable-Baselines3 (SB3) & SB3-Contrib
  * Gymnasium
  * Weights & Biases (`wandb`) for experiment tracking and metric synchronization
* **Session Persistence:** All training runs were managed via a background `tmux` virtual session (`[ssh_tmux]`), ensuring uninterrupted training across local network disconnections or client shutdowns.

---

## 3. Execution Pipeline & Scripts

The baselines were executed across four primary shell scripts:

### Step 1: Memoryless Baselines
> 📁 [`run_scripts/run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh)

* **Executed Agents:**
  * [`agents/train_dqn.py`](../agents/train_dqn.py) on `TemporalComp/HazardousDelivery-v0` (Seeds 1, 2, 3)
  * [`agents/train_dqn.py`](../agents/train_dqn.py) on `TemporalComp/SequentialColour-v0` (Seeds 1, 2, 3)
  * [`agents/train_ppo.py`](../agents/train_ppo.py) on `TemporalComp/DeferredMaintenance-v0` (Seeds 1, 2, 3)
* **Result:** All 9 runs completed successfully. As predicted by the research hypothesis, purely reactive policies failed to solve the temporal sequences.

---

### Step 2: Implicit Memory Baselines
> 📁 [`run_scripts/run_implicit_memory_baselines.sh`](../run_scripts/run_implicit_memory_baselines.sh)

* **Executed Agents:**
  * [`agents/drqn/train_drqn.py`](../agents/drqn/train_drqn.py) on `TemporalComp/HazardousDelivery-v0` (Seeds 1, 2, 3)
  * [`agents/drqn/train_drqn.py`](../agents/drqn/train_drqn.py) on `TemporalComp/SequentialColour-v0` (Seeds 1, 2, 3)
  * [`agents/train_recurrent_ppo.py`](../agents/train_recurrent_ppo.py) on `TemporalComp/DeferredMaintenance-v0` (Seeds 1, 2, 3)
* **Result:** 
  * Tabular DRQN and Recurrent PPO runs completed successfully.
  * DRQN on `SequentialColour-v0` (image observations) crashed across all 3 seeds due to image replay memory allocation constraints.

---

### Step 3: Explicit Symbolic Memory Baselines
> 📁 [`run_scripts/run_explicit_memory_baselines.sh`](../run_scripts/run_explicit_memory_baselines.sh)

* **Executed Agents:**
  * [`agents/qrm/train_qrm.py`](../agents/qrm/train_qrm.py) on `TemporalComp/HazardousDelivery-v0` (Seeds 1, 2, 3)
  * [`agents/ppo_rm/train_ppo_rm.py`](../agents/ppo_rm/train_ppo_rm.py) on `TemporalComp/DeferredMaintenance-v0` (Seeds 1, 2, 3)
* **Result:** 
  * QRM on `HazardousDelivery-v0` completed all 3 seeds.
  * PPO-RM on `DeferredMaintenance-v0` encountered runtime errors across all 3 seeds.

---

### Step 4: Remediation & Parallel Execution
> 📁 [`run_scripts/rerun_drqn_image_and_ppo_rm_continuous.sh`](../run_scripts/rerun_drqn_image_and_ppo_rm_continuous.sh)

To remediate the 6 failed runs, root causes were diagnosed via the W&B run logs and patched in the agent drivers. Taking advantage of the 40 vCPUs and 129 GB of RAM, the 6 missing runs were executed simultaneously in parallel:
* 3 parallel instances of [`agents/ppo_rm/train_ppo_rm.py`](../agents/ppo_rm/train_ppo_rm.py) on `TemporalComp/DeferredMaintenance-v0` (Seeds 1, 2, 3)
* 3 parallel instances of [`agents/drqn/train_drqn.py`](../agents/drqn/train_drqn.py) on `TemporalComp/SequentialColour-v0` (Seeds 1, 2, 3)

All 6 background jobs completed cleanly, bringing the total number of finished baseline runs to **24**.

---

## 4. Generated Outputs & Local Artifacts

### Remote Checkpoints
All trained policies and intermediate checkpoints were saved on the remote server under:
* `/workspace/temporal-composition-gym/models/` (containing `.zip` SB3 archives and `.pt` PyTorch weights).

### Live Experiment Synchronization
* Logged to Weights & Biases project: `altonmaseko-wits-university/temporal-composition-gym`
* Complete trajectories for episodic returns, episode lengths, value losses, policy losses, and hardware utilization.

### Local Exported Datasets
Data was extracted to the local repository using:
> 📁 [`export_wandb_data.py`](../export_wandb_data.py)

Saved in:
> 📁 [`wandb_phase_3_baseline_exports/`](../wandb_phase_3_baseline_exports/)

* **Summary Table:** [`wandb_phase_3_baseline_exports/all_runs_summary.csv`](../wandb_phase_3_baseline_exports/all_runs_summary.csv) — Final metrics, episode return statistics, and hyperparameter configs.
* **History Timeseries:** 24 individual `*_history.csv` files containing the raw step-by-step curve coordinates for each run.

---

## 5. Complete Registry of Baseline W&B Runs (24 Total)

| # | Paradigm | Algorithm | Environment | Seed | Run Name | Source Script |
|---|---|---|---|:---:|---|---|
| 1 | Memoryless | DQN | `HazardousDelivery-v0` | 1 | `TemporalComp_HazardousDelivery-v0__dqn__1__1790865199` | [`run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh) |
| 2 | Memoryless | DQN | `HazardousDelivery-v0` | 2 | `TemporalComp_HazardousDelivery-v0__dqn__2__1790875107` | [`run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh) |
| 3 | Memoryless | DQN | `HazardousDelivery-v0` | 3 | `TemporalComp_HazardousDelivery-v0__dqn__3__1790875136` | [`run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh) |
| 4 | Memoryless | DQN | `SequentialColour-v0` | 1 | `TemporalComp_SequentialColour-v0__dqn__1__1790866823` | [`run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh) |
| 5 | Memoryless | DQN | `SequentialColour-v0` | 2 | `TemporalComp_SequentialColour-v0__dqn__2__1790875121` | [`run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh) |
| 6 | Memoryless | DQN | `SequentialColour-v0` | 3 | `TemporalComp_SequentialColour-v0__dqn__3__1790875149` | [`run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh) |
| 7 | Memoryless | PPO | `DeferredMaintenance-v0` | 1 | `TemporalComp_DeferredMaintenance-v0__ppo__1__1790875168` | [`run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh) |
| 8 | Memoryless | PPO | `DeferredMaintenance-v0` | 2 | `TemporalComp_DeferredMaintenance-v0__ppo__2__1790882474` | [`run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh) |
| 9 | Memoryless | PPO | `DeferredMaintenance-v0` | 3 | `TemporalComp_DeferredMaintenance-v0__ppo__3__1790882487` | [`run_memoryless_baselines.sh`](../run_scripts/run_memoryless_baselines.sh) |
| 10 | Implicit | DRQN | `HazardousDelivery-v0` | 1 | `TemporalComp_HazardousDelivery-v0__drqn__1__1790926818` | [`run_implicit_memory_baselines.sh`](../run_scripts/run_implicit_memory_baselines.sh) |
| 11 | Implicit | DRQN | `HazardousDelivery-v0` | 2 | `TemporalComp_HazardousDelivery-v0__drqn__2__1790935544` | [`run_implicit_memory_baselines.sh`](../run_scripts/run_implicit_memory_baselines.sh) |
| 12 | Implicit | DRQN | `HazardousDelivery-v0` | 3 | `TemporalComp_HazardousDelivery-v0__drqn__3__1790936338` | [`run_implicit_memory_baselines.sh`](../run_scripts/run_implicit_memory_baselines.sh) |
| 13 | Implicit | Recurrent PPO | `DeferredMaintenance-v0` | 1 | `TemporalComp_DeferredMaintenance-v0__recurrent_ppo__1__1790937162` | [`run_implicit_memory_baselines.sh`](../run_scripts/run_implicit_memory_baselines.sh) |
| 14 | Implicit | Recurrent PPO | `DeferredMaintenance-v0` | 2 | `TemporalComp_DeferredMaintenance-v0__recurrent_ppo__2__1790951250` | [`run_implicit_memory_baselines.sh`](../run_scripts/run_implicit_memory_baselines.sh) |
| 15 | Implicit | Recurrent PPO | `DeferredMaintenance-v0` | 3 | `TemporalComp_DeferredMaintenance-v0__recurrent_ppo__3__1790951262` | [`run_implicit_memory_baselines.sh`](../run_scripts/run_implicit_memory_baselines.sh) |
| 16 | Explicit | QRM | `HazardousDelivery-v0` | 1 | `TemporalComp_HazardousDelivery-v0__qrm__1__1790953458` | [`run_explicit_memory_baselines.sh`](../run_scripts/run_explicit_memory_baselines.sh) |
| 17 | Explicit | QRM | `HazardousDelivery-v0` | 2 | `TemporalComp_HazardousDelivery-v0__qrm__2__1790986449` | [`run_explicit_memory_baselines.sh`](../run_scripts/run_explicit_memory_baselines.sh) |
| 18 | Explicit | QRM | `HazardousDelivery-v0` | 3 | `TemporalComp_HazardousDelivery-v0__qrm__3__1790989818` | [`run_explicit_memory_baselines.sh`](../run_scripts/run_explicit_memory_baselines.sh) |
| 19 | Implicit | DRQN | `SequentialColour-v0` | 1 | `TemporalComp_SequentialColour-v0__drqn__1__1791049667` | [`rerun_drqn_image_and_ppo_rm_continuous.sh`](../run_scripts/rerun_drqn_image_and_ppo_rm_continuous.sh) |
| 20 | Implicit | DRQN | `SequentialColour-v0` | 2 | `TemporalComp_SequentialColour-v0__drqn__2__1791049667` | [`rerun_drqn_image_and_ppo_rm_continuous.sh`](../run_scripts/rerun_drqn_image_and_ppo_rm_continuous.sh) |
| 21 | Implicit | DRQN | `SequentialColour-v0` | 3 | `TemporalComp_SequentialColour-v0__drqn__3__1791049667` | [`rerun_drqn_image_and_ppo_rm_continuous.sh`](../run_scripts/rerun_drqn_image_and_ppo_rm_continuous.sh) |
| 22 | Explicit | PPO-RM | `DeferredMaintenance-v0` | 1 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__1__1791049669` | [`rerun_drqn_image_and_ppo_rm_continuous.sh`](../run_scripts/rerun_drqn_image_and_ppo_rm_continuous.sh) |
| 23 | Explicit | PPO-RM | `DeferredMaintenance-v0` | 2 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__2__1791049669` | [`rerun_drqn_image_and_ppo_rm_continuous.sh`](../run_scripts/rerun_drqn_image_and_ppo_rm_continuous.sh) |
| 24 | Explicit | PPO-RM | `DeferredMaintenance-v0` | 3 | `TemporalComp_DeferredMaintenance-v0__ppo_rm__3__1791049669` | [`rerun_drqn_image_and_ppo_rm_continuous.sh`](../run_scripts/rerun_drqn_image_and_ppo_rm_continuous.sh) |

---

## 6. Key Empirical Takeaways

> [!IMPORTANT]
> **Core Hypothesis Validated:** Purely reactive, memoryless models (DQN, PPO) are structurally incapable of resolving delayed temporal dependencies, validating the benchmark's non-Markovian design.

* **Early Learning Signals:** Both explicit memory (QRM) and implicit recurrent memory (DRQN) successfully identified the sequential goals early in training, with QRM hitting optimal peak returns of **+80.0** on `HazardousDelivery-v0`.
* **Catastrophic Forgetting:** Without stabilization, both architectures suffered from severe policy collapse during later training stages.
* **Next Steps:** These observations directly established the objective for hyperparameter tuning: optimizing learning rate schedules, target network update frequencies, recurrent sequence lengths (BPTT), and entropy regularization to lock in policy stability.
