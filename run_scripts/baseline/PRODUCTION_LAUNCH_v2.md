# Protocol v2 baseline: production launch and collection contract

Status: **ready to launch, subject to the cloud preflight on each machine** (step B in sections 4-6). Validated code revision: `7d94a93`.
All agreed comparisons and local checks are implemented; the local offline validation
report is [validation/pipeline_validation_v2.json](validation/pipeline_validation_v2.json) and
the task solvability report is [validation/task_validation_v2.json](validation/task_validation_v2.json).

## 1. Shared revision

Launch every machine from the same commit of branch `new_experiments`. The validated code
revision is named in section 8; later commits that change only documentation or reports
are equivalent (the setup step checks that the code is identical). Every run records the exact
`git rev-parse HEAD`, dirty-file list and diff hash in its `metadata.json`, and preflight warns
if the checkout is not clean. The machines clone `new_experiments` from GitHub.

## 2. What will run

| | memoryless machine | implicit machine | explicit machine |
|---|---|---|---|
| Hazardous Delivery (1,000,000 steps) | DQN | DRQN | QRM |
| Sequential Colour (3,000,000 steps) | DQN | DRQN | QRM-Cue |
| Deferred Maintenance (2,048,000 steps) | PPO | Recurrent PPO | PPO-RM |
| Controls (after the main runs) | DQN-TS delivery, DQN-TS colour | – | – |
| Runs | 25 | 15 | 15 |

Seeds 1, 2, 3, 4, 5 (6-10 reserved for the pre-registered expansion rule). Order:
seed-major (all of seed 1, then seed 2, ...), one process at a time. Total **55 runs**.
Evaluation: step 0 and every 5% of the budget on episode seeds 1,000,000+ (50 episodes; 20
for maintenance), final evaluation on episode seeds 2,000,000+ (200; 100 for maintenance),
greedy policy. Full resolved configurations and config hashes: [baseline_plan.json](baseline_plan.json).

W&B: project `new_experiments` (default entity of the logged-in account unless
`--wandb-entity` is given), group `baseline-v2`, job type `<env>/<algo>`, run name = run id,
tags `baseline, v2, full, <family>, <env>, <algo>, seed<n>`. Smoke/preflight runs use group
`baseline-v2-smoke` / `preflight-v2`. The x-axis of every metric is `global_step`
(environment interactions).

## 3. The three machines

| Role | Vast.ai instance | Connect from PowerShell on your laptop | Runs |
|---|---|---|---|
| memoryless | 57966532 | `ssh -p 43164 root@85.146.9.40 -L 8080:localhost:8080` | 25 |
| implicit | 57966535 | `ssh -p 40002 root@142.198.191.106 -L 8080:localhost:8080` | 15 |
| explicit | 57966540 | `ssh -p 40031 root@83.229.2.69 -L 8080:localhost:8080` | 15 |

Each SSH login opens inside tmux on the machine, so training keeps running when you close
PowerShell; connecting again with the same command returns you to it. If you have two or three
connections open at once, the second and third print a harmless warning that local port 8080 is
already in use.

Every machine follows the same five steps: **A** setup, **B** preflight, **C** launch, **D** wait,
**E** bundle the results. Then you collect each machine from your laptop (section 7a) and publish once all three are collected (section 7b). The
commands below are complete; run them exactly as written, one block at a time, on the machine
named in the heading.

## 4. Memoryless machine (instance 57966532)

Connect (PowerShell on your laptop):
```powershell
ssh -p 43164 root@85.146.9.40 -L 8080:localhost:8080
```

**A. Setup** (once). Clone the code:
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```
Confirm the code equals the validated revision. It must print the commit line, then `CODE CHECK DONE`, and nothing in between:
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --stat 7d94a93 HEAD -- '*.py' '*.sh' requirements.txt setup.py run_scripts/baseline/baseline_plan.json ':(exclude)play_human' && echo "CODE CHECK DONE"
```
Create the Python environment (installs the venv package first if the image lacks it):
```bash
python3 --version && (python3 -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -m venv --clear ~/tcg))
```
Install the dependencies (several minutes; PyTorch is large):
```bash
~/tcg/bin/python -m pip install --upgrade pip && ~/tcg/bin/python -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in to W&B (paste your API key when asked; the input stays hidden):
```bash
~/tcg/bin/wandb login
```

**B. Preflight** (3-10 minutes). It must end with `PREFLIGHT PASSED`:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python run_scripts/baseline/preflight.py --role memoryless --learning-check --out ~/preflight_memoryless.json
```

**C. Launch** the 25 runs:
```bash
cd ~/temporal-composition-gym && bash run_scripts/baseline/run_memoryless_baselines.sh --python ~/tcg/bin/python --threads 4 2>&1 | tee -a ~/baseline_memoryless.log
```

**D. Wait.** You can close PowerShell. To check progress later, connect again, or from any shell on the machine run:
```bash
tail -n 20 ~/baseline_memoryless.log
```
It is finished when the log ends with `All 25 runs exited 0 (each verified its budget and final evaluation).` If it ends with `BATCH FAILED` instead, see section 6.

**E. Bundle the results** (after it has finished). It shows the end of the log, adds the preflight report and log to the results, packs everything except the restart checkpoints (`checkpoints/restart`, which hold the replay buffer) and the W&B cache, and prints the bundle size:
```bash
cd ~/temporal-composition-gym && tail -n 3 ~/baseline_memoryless.log && mkdir -p run_scripts/baseline/results/v2/full/_machine_logs/memoryless && cp ~/preflight_memoryless.json ~/baseline_memoryless.log run_scripts/baseline/results/v2/full/_machine_logs/memoryless/ && tar -czf ~/collect_memoryless.tar.gz --exclude='*/checkpoints/restart*' --exclude='*/wandb' run_scripts/baseline/results/v2/full && ls -lh ~/collect_memoryless.tar.gz
```

## 5. Implicit-memory machine (instance 57966535)

Connect (PowerShell on your laptop):
```powershell
ssh -p 40002 root@142.198.191.106 -L 8080:localhost:8080
```

**A. Setup** (once). Clone the code:
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```
Confirm the code equals the validated revision. It must print the commit line, then `CODE CHECK DONE`, and nothing in between:
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --stat 7d94a93 HEAD -- '*.py' '*.sh' requirements.txt setup.py run_scripts/baseline/baseline_plan.json ':(exclude)play_human' && echo "CODE CHECK DONE"
```
Create the Python environment:
```bash
python3 --version && (python3 -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -m venv --clear ~/tcg))
```
Install the dependencies:
```bash
~/tcg/bin/python -m pip install --upgrade pip && ~/tcg/bin/python -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in to W&B:
```bash
~/tcg/bin/wandb login
```

**B. Preflight.** It must end with `PREFLIGHT PASSED`:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python run_scripts/baseline/preflight.py --role implicit --learning-check --out ~/preflight_implicit.json
```

**C. Launch** the 15 runs:
```bash
cd ~/temporal-composition-gym && bash run_scripts/baseline/run_implicit_memory_baselines.sh --python ~/tcg/bin/python --threads 4 2>&1 | tee -a ~/baseline_implicit.log
```

**D. Wait.** Check progress:
```bash
tail -n 20 ~/baseline_implicit.log
```
It is finished when the log ends with `All 15 runs exited 0 (each verified its budget and final evaluation).`

**E. Bundle the results:**
```bash
cd ~/temporal-composition-gym && tail -n 3 ~/baseline_implicit.log && mkdir -p run_scripts/baseline/results/v2/full/_machine_logs/implicit && cp ~/preflight_implicit.json ~/baseline_implicit.log run_scripts/baseline/results/v2/full/_machine_logs/implicit/ && tar -czf ~/collect_implicit.tar.gz --exclude='*/checkpoints/restart*' --exclude='*/wandb' run_scripts/baseline/results/v2/full && ls -lh ~/collect_implicit.tar.gz
```

## 6. Explicit-memory machine (instance 57966540)

Connect (PowerShell on your laptop):
```powershell
ssh -p 40031 root@83.229.2.69 -L 8080:localhost:8080
```

**A. Setup** (once). Clone the code:
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```
Confirm the code equals the validated revision. It must print the commit line, then `CODE CHECK DONE`, and nothing in between:
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --stat 7d94a93 HEAD -- '*.py' '*.sh' requirements.txt setup.py run_scripts/baseline/baseline_plan.json ':(exclude)play_human' && echo "CODE CHECK DONE"
```
Create the Python environment:
```bash
python3 --version && (python3 -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -m venv --clear ~/tcg))
```
Install the dependencies:
```bash
~/tcg/bin/python -m pip install --upgrade pip && ~/tcg/bin/python -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in to W&B:
```bash
~/tcg/bin/wandb login
```

**B. Preflight.** It must end with `PREFLIGHT PASSED`:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python run_scripts/baseline/preflight.py --role explicit --learning-check --out ~/preflight_explicit.json
```

**C. Launch** the 15 runs:
```bash
cd ~/temporal-composition-gym && bash run_scripts/baseline/run_explicit_memory_baselines.sh --python ~/tcg/bin/python --threads 4 2>&1 | tee -a ~/baseline_explicit.log
```

**D. Wait.** Check progress:
```bash
tail -n 20 ~/baseline_explicit.log
```
It is finished when the log ends with `All 15 runs exited 0 (each verified its budget and final evaluation).`

**E. Bundle the results:**
```bash
cd ~/temporal-composition-gym && tail -n 3 ~/baseline_explicit.log && mkdir -p run_scripts/baseline/results/v2/full/_machine_logs/explicit && cp ~/preflight_explicit.json ~/baseline_explicit.log run_scripts/baseline/results/v2/full/_machine_logs/explicit/ && tar -czf ~/collect_explicit.tar.gz --exclude='*/checkpoints/restart*' --exclude='*/wandb' run_scripts/baseline/results/v2/full && ls -lh ~/collect_explicit.tar.gz
```

## 6b. Monitoring, failures and restart

- **Per run**: the launcher prints `[i/N] env / algo / seed s`, then `ok: complete: ...` or
  `FAILED (exit N)` with the last lines of stderr. Per-run logs:
  `~/temporal-composition-gym/run_scripts/baseline/results/v2/full/_batches/<batch>/logs/`.
- **Per batch**: `batch_status.tsv` in the same batch folder (exit code per run). The launcher
  exits nonzero and prints `BATCH FAILED` if any run failed or was not attempted; it never prints a
  success summary then.
- **Per run directory**: `status.json` `state` is `complete` only after the budget and every
  evaluation are verified; `failed` carries the error and traceback.
- **Whole matrix**: `check_manifest.py` (section 7) exits 0 only when all 55 runs are complete
  exactly once.

**If a machine reboots, the connection to a run is lost, or the log ends with `BATCH FAILED`:**
connect to that machine and run its restart command. It skips runs that are already complete,
resumes unfinished runs from their last checkpoint (every 10% of the budget), and starts the rest
fresh. Do not rerun the plain launch command: that would start every run again.

Memoryless machine:
```bash
cd ~/temporal-composition-gym && bash run_scripts/baseline/run_memoryless_baselines.sh --python ~/tcg/bin/python --threads 4 --continue-batch 2>&1 | tee -a ~/baseline_memoryless.log
```
Implicit-memory machine:
```bash
cd ~/temporal-composition-gym && bash run_scripts/baseline/run_implicit_memory_baselines.sh --python ~/tcg/bin/python --threads 4 --continue-batch 2>&1 | tee -a ~/baseline_implicit.log
```
Explicit-memory machine:
```bash
cd ~/temporal-composition-gym && bash run_scripts/baseline/run_explicit_memory_baselines.sh --python ~/tcg/bin/python --threads 4 --continue-batch 2>&1 | tee -a ~/baseline_explicit.log
```

If the same run fails again, look at the error in its `status.json` (the `FAILED` line names the
run) before retrying. Resume is an episode-boundary restart: weights, optimiser, replay, RNG
streams, counters and evaluation progress are restored and the episode in progress starts again;
`status.json` reports `restored_steps` and `new_steps_this_session`.

## 7. Download, check and publish

Do **7a** for each machine as soon as its step E is done, in any order; that machine can then be
destroyed even if the others are still running. Do **7b** once, after 7a has been done for all
three machines. Every command uses full paths, so it works from any PowerShell window.

### 7a. Collect one finished machine

**Memoryless machine (instance 57966532, 25 runs)**

1. On the machine, print the bundle's fingerprint:
   ```bash
   sha256sum ~/collect_memoryless.tar.gz
   ```
2. On your laptop (PowerShell), download the bundle:
   ```powershell
   scp -P 43164 root@85.146.9.40:collect_memoryless.tar.gz "$HOME\Downloads"
   ```
3. Print the downloaded copy's fingerprint. It must equal the one from step 1:
   ```powershell
   (Get-FileHash "$HOME\Downloads\collect_memoryless.tar.gz").Hash.ToLower()
   ```
4. Unpack it into the repository:
   ```powershell
   tar -xzf "$HOME\Downloads\collect_memoryless.tar.gz" -C "C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym"
   ```
5. Check this machine's runs. It must print `25`:
   ```powershell
   (D:\miniconda3\envs\temporal_gym\python.exe "C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym\run_scripts\baseline\check_manifest.py" | Select-String "^complete\s+memoryless\s").Count
   ```
6. Copy `collect_memoryless.tar.gz` from Downloads to your backup location (external disk or cloud drive).
7. Only if steps 3 and 5 passed, destroy instance 57966532 in the Vast.ai console.

**Implicit machine (instance 57966535, 15 runs)**

1. On the machine:
   ```bash
   sha256sum ~/collect_implicit.tar.gz
   ```
2. On your laptop:
   ```powershell
   scp -P 40002 root@142.198.191.106:collect_implicit.tar.gz "$HOME\Downloads"
   ```
3. Must equal step 1:
   ```powershell
   (Get-FileHash "$HOME\Downloads\collect_implicit.tar.gz").Hash.ToLower()
   ```
4. Unpack:
   ```powershell
   tar -xzf "$HOME\Downloads\collect_implicit.tar.gz" -C "C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym"
   ```
5. Must print `15`:
   ```powershell
   (D:\miniconda3\envs\temporal_gym\python.exe "C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym\run_scripts\baseline\check_manifest.py" | Select-String "^complete\s+implicit\s").Count
   ```
6. Back up `collect_implicit.tar.gz`.
7. Only if steps 3 and 5 passed, destroy instance 57966535.

**Explicit machine (instance 57966540, 15 runs)**

1. On the machine:
   ```bash
   sha256sum ~/collect_explicit.tar.gz
   ```
2. On your laptop:
   ```powershell
   scp -P 40031 root@83.229.2.69:collect_explicit.tar.gz "$HOME\Downloads"
   ```
3. Must equal step 1:
   ```powershell
   (Get-FileHash "$HOME\Downloads\collect_explicit.tar.gz").Hash.ToLower()
   ```
4. Unpack:
   ```powershell
   tar -xzf "$HOME\Downloads\collect_explicit.tar.gz" -C "C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym"
   ```
5. Must print `15`:
   ```powershell
   (D:\miniconda3\envs\temporal_gym\python.exe "C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym\run_scripts\baseline\check_manifest.py" | Select-String "^complete\s+explicit\s").Count
   ```
6. Back up `collect_explicit.tar.gz`.
7. Only if steps 3 and 5 passed, destroy instance 57966540.

Destroying an instance at the end of its 7a is fine: none of the commands in 7b run on the Vast.ai
machines. 7b only uses the files already on your laptop. Its bundle has been verified, unpacked,
checked and backed up, and its W&B data is already online; only its restart checkpoints are lost,
and those are needed only to resume unfinished runs. Repeating a machine's 7a is harmless:
unpacking again overwrites identical files, so runs are never duplicated.

### 7b. Publish (once, after 7a has been done for all three machines)

Go to the repository and bring it up to date with GitHub:
```powershell
cd "C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym"
```
```powershell
git pull --ff-only origin new_experiments
```

Check the full matrix. The last line must read `55/55 planned runs complete`:
```powershell
D:\miniconda3\envs\temporal_gym\python.exe run_scripts/baseline/check_manifest.py --json run_scripts/baseline/results/v2/full/manifest.json
```

Stage the results and confirm that no checkpoint files are staged (this second command must print nothing):
```powershell
git add run_scripts/baseline/results
```
```powershell
git status --short | Select-String checkpoints
```

Commit and push:
```powershell
git commit -m "Add protocol v2 baseline results from the three machines"
```
```powershell
git push origin new_experiments
```

GitHub then holds a single results commit containing the complete, verified set of 55 runs.

What ends up where:

| | Laptop | GitHub | W&B |
|---|---|---|---|
| CSV/JSON records, manifest, batch logs, machine logs | yes | yes | – |
| Policy snapshots (`checkpoints/policy_step*`, every 10%) | yes | no (ignored by `.gitignore`) | – |
| Restart checkpoints with replay (`checkpoints/restart`) | no | no | – |
| Metrics, curves, summaries | in the CSVs | in the CSVs | yes, project `new_experiments` |

Until the push in 7b, your laptop and your backup hold the only copies of the results of any
destroyed instance, and the policy snapshots stay only there for good, so keep the backed-up
`.tar.gz` files.

**Cloud-specific checks that are still open** (they cannot be done locally):

1. Online W&B login and upload from a training process (preflight `wandb online run` and
   `learning check ... W&B https://...`).
2. CUDA operation of the colour runs and the GPU's VRAM (preflight `cuda`, `vram`, `learning check colour/...`).
3. Linux `bash` running the launchers (preflight is started by them; `--dry-run` shows the commands).
4. Installation of `requirements.txt` on the machine image (preflight `package ...`).
5. Real throughput and total wall-clock (preflight `rough full-run time`, then the first
   evaluation lines). Local CPU-only smoke throughput suggests colour runs are the critical path.
6. Disk use during the runs (each colour restart checkpoint holds the replay: about 4.3 GB for
   DQN, 2.4 GB for DRQN, 2.1 GB for QRM-Cue, replaced at each checkpoint).

## 8. Validation summary

**Validated code revision: `7d94a93`** (branch `new_experiments`). Launch from that commit or any
later commit that only changes documentation or the validation reports. Both reports were produced
from `7d94a93`; their `dirty` flag is set only because the reports themselves had been rewritten
(`changed_tracked_files` lists just `validation/task_validation_v2.json`).

Local machine: Windows 11, Python 3.10.20, CPU only (no CUDA), W&B offline.

**Pipeline validation** ([validation/pipeline_validation_v2.json](validation/pipeline_validation_v2.json)):
115 of 115 checks passed in 14.5 minutes.

| Section | What was shown |
|---|---|
| matrix (11 runs) | every environment/algorithm combination, including both DQN-TS controls, completed a labelled smoke run with real gradient updates, 3 evaluation points, 3 final-evaluation episodes, exact budget accounting (training-episode steps never exceed the budget; evaluation steps are separate), finite losses and returns, a restart checkpoint whose reloaded weights equal the final policy snapshot (plus replay for off-policy agents), and offline W&B records that, decoded, match the local files |
| resume (5 algorithms) | DQN, PPO, Recurrent PPO, DRQN and QRM: injected failure recorded as `failed`; resumed run completed with restored + new steps = budget, the **same update count as the uninterrupted run** (875, 16, 16, 876, 876), no duplicated evaluation rows, and the same step-0 evaluation |
| collision | a second seed and a second configuration got separate directories, run ids, W&B ids and config hashes; the second configuration started from an untrained policy (identical step-0 evaluation) with its full update count |
| failure | a deliberate failure is `failed` in status and manifest, has no summary and a nonzero exit; removing `eval_final.csv`, the last evaluation point or `final_summary.json` makes the run unverified and uncounted; an evaluation with no episodes raises instead of producing zeros |
| config | rejected: unknown override, seed override, pilot W&B project, tuning-range seed, a maintenance budget SB3 would round, an algorithm outside the plan, an impossible readiness range; `baseline_plan.json` matches the code |
| envs | random rollouts in all three tasks: finite observations inside their spaces, finite rewards, well-formed events and termination info |
| memory | projected full-run image replay: DQN/DQN-TS colour 4.24 GB (SB3 stores obs and next obs), DRQN colour 2.32 GB (frames + LSTM state), QRM-Cue colour 2.12 GB; smoke colour runs peaked at 0.5-0.6 GB RSS |
| manifest | all 11 smoke combinations `complete` |
| pytest | 75 semantic and pipeline tests passed |

Settings that differ in smoke mode (recorded per run in the report under `smoke_vs_full`, and in each
run's resolved config): budgets (4,000 / 3,000 / 4,096 env steps), 2 evaluation points with 2 periodic
and 3 final episodes, checkpoints every half budget, training-log interval, maintenance episode limit
600 steps, and shortened schedules: learning starts 500, target update 250, replay 5,000 frames, epsilon
decay over 50% (DQN-family), PPO `n_steps` 512 and 2 epochs, DRQN colour 2 sequences of 16 steps with
8-step burn-in, QRM batch 8. W&B group `baseline-v2-smoke`. Production settings are unchanged.

**Task validation** ([validation/task_validation_v2.json](validation/task_validation_v2.json)), 200
final-evaluation seeds per task, with the reward machine checked against the env on every step:

| Task | Oracle success | Oracle mean return | Oracle p95 length / limit | Random success | Random mean return |
|---|---|---|---|---|---|
| Delivery | 100% | 80.0 | 46 / 500 | 0% | -49.5 |
| Colour | 100% | 10.0 | 92 / 300 | 0% | -0.8 |
| Maintenance | 100% | 1076.2 | 1,859 / 4,000 | 0% | -6,037.1 |

**Throughput hint (local CPU only, smoke settings):** colour runs are the slowest (QRM-Cue ~76 env
steps/s on this laptop's CPU, about 11 h per full run). They run on the GPU in production; the cloud
preflight reports measured projections for each machine.

**Readiness:** the agreed comparisons (45 main runs + 10 controls) and all local checks are
implemented and passing, so the pipeline is ready for production **once each machine's cloud
preflight passes** (the list after section 7 shows what only the cloud can verify).

## 9. Baseline collection contract (what the analysis phase receives)

Input: one or more results roots containing `v2/full/<env>/<algo>/seed<s>__<cfghash10>__<uid>/`
run directories, plus `manifest.json` from `check_manifest.py --json`. Use only runs whose
manifest state is `complete`; `failed`, `missing`, `duplicate` and `nonstandard` runs must be
reported, never imputed as zero. All files are UTF-8; CSVs have a header row; empty cells mean
"not applicable/not measured", never zero.

| File | Content and fields |
|---|---|
| `manifest.json` | `protocol_version`, `mode`, `roots`, `expected` (55), `counts`, `all_complete`, `entries[]` = planned run {`role`, `env`, `algo`, `seed`, `family` (memoryless/implicit/explicit/control), `script`, `task_state`, `config_hash` (planned), `state`, `attempts[]` {`run_dir`, `run_id`, `config_hash`, `state`, `verified`, `problems`}}, `unexpected_runs[]` |
| `config.resolved.json` | full resolved configuration: `protocol_version`, `phase`, `mode`, `env` {`key`, `id`, `variant`, `kwargs`}, `algo` {`name`, `family`, `impl`, `hyperparams`}, `seed` {`train_seed`, `train_env_reset_seed`}, `budget.env_steps`, `eval` {`n_points`, `every_steps`, `periodic_episodes`, `final_episodes`, seed blocks, `policy`}, `checkpoint`, `logging`, `overrides`, `wandb` {`project`, `entity`, `group`} |
| `identity.json` | `run_id`, `run_uid`, `wandb_id`, `config_hash` (sha256 of the config without `wandb`), `env`, `algo`, `family`, `seed`, `mode`, `machine_role`, `batch_id`, `run_dir`, `created_utc` |
| `metadata.json` (+ `metadata_resume_*.json`) | `git` {`rev`, `branch`, `dirty`, `changed_tracked_files`, `diff_sha256`}, `python`, `packages`, `hardware` (host, CPU, RAM, disk, GPUs, `nvidia_smi`), `device` {`requested`, `selected`, `is_gpu_run`, `gpu_name`}, `threads`, `env_vars` (allow-listed), `argv`, `launcher_logs` |
| `status.json` | `state` (complete/failed/interrupted/running), `verified`, `budget`, `env_steps_done`, `n_updates`, `final_eval_done`, `restored_steps`, `new_steps_this_session`, `resumes[]`, `wandb` {`id`, `url`, `mode`, `project`, `entity`}, timestamps, `error`, `traceback` |
| `final_summary.json` | `global_step`, `n_updates`, `train_episodes`, `final_eval` {`n_episodes`, `return_mean/std/min/max`, `success_rate`, `length_mean`, `progress_mean`, `causes`}, `last_periodic_eval` (same fields), `config_hash`, `run_id`, `completed_utc` |
| `eval_points.csv` | one row per periodic evaluation (21): `eval_index`, `global_step` (0, 5%, ..., budget), `n_episodes`, `return_mean`, `return_std`, `return_min`, `return_max`, `success_rate`, `length_mean`, `progress_mean`, `causes_json`, `n_updates` |
| `eval_periodic.csv` | every periodic evaluation episode: `eval_index`, `global_step`, `episode_seed` (1,000,000+i, identical across algorithms), `return_raw`, `length`, `success`, `cause`, `progress` |
| `eval_final.csv` | every final evaluation episode at the budget: `global_step`, `episode_seed` (2,000,000+i), `return_raw`, `length`, `success`, `cause`, `progress` |
| `train_episodes.csv` | every completed training episode: `global_step` (at its end), `episode`, `return_raw` (env reward), `return_rm` (RM reward; explicit agents and DQN-TS, equal to `return_raw`), `length`, `success`, `cause`, `progress` |
| `train_updates.csv` | aggregated every 0.5% of the budget: `global_step`, `n_updates` (cumulative gradient updates), `epsilon`, `loss`, `q_mean`, `grad_norm`, `extra_json` (SB3 logger values or DRQN/QRM extras, e.g. `valid_heads_per_transition`) |
| `resources.csv` | at each evaluation: `global_step`, `wall_time_s`, `env_steps_per_s`, `rss_gb`, `gpu_mem_alloc_gb`, `gpu_mem_max_gb`, `replay_frames` |
| `checkpoints/` | `policy_step<N>.zip` (SB3) / `.pt` (DRQN/QRM Q-network) every 10%; `restart/` latest full restart state |
| `wandb/` | W&B's local copy of the run (same numbers as the CSVs) |
| batch folder `_batches/<role>__<host>__<utc>-<n>/` | `batch_status.tsv` (`index`, `env`, `algo`, `seed`, `exit_code`, `started_utc`, `finished_utc`, `stdout`, `stderr`), `preflight.json`, `logs/` |

Field meanings: `success` is the task's own success flag (delivery: goal after all deliveries;
colour: full sequence; maintenance: final exit with every code serviced). `cause` is one of the
env's termination causes (`success`, `lava`, `damage_death`, `premature_goal`,
`wrong_wall_death`, `uncollected_codes_at_exit`, `held_codes_at_final_exit`, `timeout`).
`progress` counts deliveries, correct walls, or services. Returns are raw environment returns;
normalise with the oracle and random references in `validation/task_validation_v2.json`.

The next phase reconciles the manifest, verifies budgets and independent seeds, inspects the
learning and evaluation records and their variability, and only then ranks algorithms; tuning
design follows from those diagnostics.
