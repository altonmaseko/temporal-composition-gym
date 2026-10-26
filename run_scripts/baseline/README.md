# Protocol v2 baseline runs: launch, restart and collection

The frozen protocol is [BASELINE_PROTOCOL.md](BASELINE_PROTOCOL.md); task changes are in
[TASK_CHANGES_v2.md](TASK_CHANGES_v2.md). The resolved, machine-readable plan (every run's
full configuration and config hash) is [baseline_plan.json](baseline_plan.json), written by
`python -m agents.common.baseline_plan write`.

## What runs where

| Machine (role) | Launcher | Runs, in order (seed-major, seeds 1-5) | Count |
|---|---|---|---|
| memoryless | `run_memoryless_baselines.sh` | DQN delivery, DQN colour, PPO maintenance; then DQN-TS delivery, DQN-TS colour (controls) | 25 |
| implicit | `run_implicit_memory_baselines.sh` | DRQN delivery, DRQN colour, Recurrent PPO maintenance | 15 |
| explicit | `run_explicit_memory_baselines.sh` | QRM delivery, QRM-Cue colour, PPO-RM maintenance | 15 |

Budgets (environment steps, identical for every algorithm in an environment): delivery
1,000,000; colour 3,000,000; maintenance 2,048,000 (= 1,000 PPO rollouts of 2,048, so SB3
never rounds). Each run evaluates at step 0 and every 5% of its budget, then runs the final
evaluation on 200 (maintenance: 100) held-out episodes. One training process at a time.

Device policy (`--device-policy plan`, default): colour runs use CUDA and fail rather than
fall back to the CPU; delivery and maintenance runs use the CPU (small MLPs dominated by
simulation). The selected device is recorded in every run's `metadata.json`.

## One-time setup on each rented machine

```bash
git clone https://github.com/altonmaseko/temporal-composition-gym.git && cd temporal-composition-gym
git checkout <the agreed commit on new_experiments>
python -m venv ~/tcg && ~/tcg/bin/python -m pip install -r requirements.txt   # pick the CUDA torch build for the machine
wandb login              # paste your API key once; W&B stores it in ~/.netrc
```

The code never reads, requires, prints or stores the API key and needs no `.env` file. The
W&B project defaults to `new_experiments` (`--wandb-project`, `--wandb-entity` to change);
the old pilot project `temporal-composition-gym` is refused.

## Preflight (before any training)

```bash
~/tcg/bin/python run_scripts/baseline/preflight.py --role implicit --learning-check
```

It checks the interpreter, that this checkout's code is imported, package versions against
`requirements.txt`, CUDA/VRAM, RAM and disk for the role's largest run, that W&B is logged in
and can create an online run in the project, and (with `--learning-check`) a short smoke
training of each of the role's algorithms. It installs nothing. The launchers run the same
preflight automatically (without the learning check) and stop before training if it fails.

## Launch

```bash
tmux new -s baseline
bash run_scripts/baseline/run_implicit_memory_baselines.sh --python ~/tcg/bin/python
```

Useful options (`--help` for all): `--threads N` (default 4), `--seeds "1 2 3"`,
`--stop-on-failure`, `--dry-run` (print the exact commands), `--mode smoke` (short labelled
check, outputs under `results/v2/smoke/`, never mixed into production).

A failed run is shown immediately with the tail of its stderr, recorded in the batch's
`batch_status.tsv`, and the batch continues with the next run (unless `--stop-on-failure`).
The launcher exits nonzero and prints `BATCH FAILED` if any run failed or was not attempted.

## Outputs

```
run_scripts/baseline/results/v2/full/
  _batches/<role>__<host>__<utc>-<n>/      batch_status.tsv, preflight.json, logs/<i>_<env>__<algo>__s<seed>.{out,err}
  <env>/<algo>/seed<s>__<cfghash10>__<utc>-<rand>/
      config.resolved.json identity.json metadata.json status.json final_summary.json
      train_episodes.csv train_updates.csv resources.csv
      eval_points.csv eval_periodic.csv eval_final.csv
      checkpoints/policy_step*.{zip,pt}   checkpoints/restart/   wandb/
```

`status.json` `state`: `running`, `complete` (budget verified, all evaluations written),
`failed` (error and traceback recorded), `interrupted`. Only `complete` + `verified` runs count.
Every directory name contains a fresh random suffix, so runs never overwrite each other.

## Restart a failed or interrupted run

After a crash or reboot, rerun the machine's launcher command with `--continue-batch`: planned
runs that are already complete are skipped, unfinished runs with a restart checkpoint are
resumed, and the rest start fresh (`run_scripts/baseline/batch_action.py` makes the decision).
Without it the launcher always starts fresh runs, which the manifest would flag as duplicates.

To resume one run by hand, re-run the same entry point with the same `--env`, `--seed` and
`--mode`, and `--resume`:

```bash
~/tcg/bin/python agents/drqn/train_drqn.py --env colour --seed 3 --device cuda --require-gpu \
    --resume run_scripts/baseline/results/v2/full/colour/drqn/seed3__<hash>__<uid>
```

The runner refuses if the configuration hash, seed, environment or algorithm differ, if the run
is already complete, or if there is no restart checkpoint. Restart checkpoints are written every
10% of the budget (SB3 at rollout boundaries). Resume is an **episode-boundary restart**: weights,
target network, optimiser, update counters, exploration schedule, replay (off-policy), RNG streams
and evaluation progress are restored; the episode that was in progress starts afresh. It is not a
bit-exact continuation. `status.json` records `restored_steps` and `new_steps_this_session`;
CSV rows written after the checkpoint are moved to `*.pre-resume-*.bak`. W&B continues the same run.

## Collect and merge the three machines

Run directories are unique, so plain copies merge without overwriting:

```bash
rsync -a memoryless-host:temporal-composition-gym/run_scripts/baseline/results/v2/full/ results_all/v2/full/
rsync -a implicit-host:temporal-composition-gym/run_scripts/baseline/results/v2/full/   results_all/v2/full/
rsync -a explicit-host:temporal-composition-gym/run_scripts/baseline/results/v2/full/   results_all/v2/full/
python run_scripts/baseline/check_manifest.py --root results_all --json results_all/manifest.json
```

(`--root` can also be given once per machine copy without merging.) The manifest marks each of
the 55 planned runs `complete`, `duplicate`, `nonstandard` (config hash differs from the plan),
`failed` or `missing`, and exits 0 only when every planned run is complete exactly once.
Official numbers come from the local CSV/JSON files, never from W&B exports or stdout.

## Local checks

```bash
python -m pytest                      # semantic + pipeline tests (about 10 minutes)
python run_scripts/baseline/validate_tasks.py   # oracle / random solvability report
```
