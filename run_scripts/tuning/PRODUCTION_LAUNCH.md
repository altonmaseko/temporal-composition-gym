# Tuning v3 launch and collection

The frozen [protocol](TUNING_PROTOCOL.md) and checked-in `trials_v3.json` are authoritative. Production is 88 screening runs followed by 66 fresh confirmation runs. Use three user-started machines, roles **a/b/c**, one process each. No script provisions machines or supplies credentials. Smoke is a plumbing check only: 64 steps, 32-step episode caps, two candidates per cell, then six confirmation runs per cell (88 tiny runs). Its output is never eligible for the scientific comparison.

## 1. Setup and plan check

Mirror `../baseline/PRODUCTION_LAUNCH_v2.md`: prepare the user-managed environment and checkout, install that checkout's requirements, record its revision, and authenticate W&B yourself on each cloud machine. Do not install or authenticate on the laptop as part of launch. Use explicit interpreter paths throughout; Windows uses `D:\miniconda3\envs\temporal_gym\python.exe -B` only. No baseline results or checkpoints are inputs to tuning execution.

```bash
PY=/absolute/path/to/environment/bin/python
"$PY" -B run_scripts/tuning/generate_trials.py --check
"$PY" -B -m agents.common.baseline_plan write --check
```

## 2. Preflight

Run on each assigned cloud machine, substituting its role. It reuses baseline interpreter/import, package, revision, CUDA compute/VRAM, RAM/disk and W&B login/online-run checks. The learning check runs real updates and both evaluations for every cell assigned to the role. Full launch requires colour on CUDA. Use offline/disabled and CPU for local diagnostics only.

```bash
"$PY" -B run_scripts/tuning/preflight.py --role a --device-policy plan --wandb-mode online --learning-check
```

Preflight failures must be resolved before launch. Record actual machine billing start **before** setup/preflight. Allowances are A 33.34, B 33.33, C 33.33 hours; at $0.30/h these sum to the agreed 100 hours/$30. A higher `--hourly-rate` reduces time allowance. Include earlier rental sessions in `--previously-billed-hours`. Terms are saved and cannot silently reset within the same results root. Reconcile the preserved ledger before any new rental/root. The launcher pauses when a conservative next-run projection threatens its allowance, and terminates an overlong child five minutes before its deadline. It cannot stop rental billing: **stop idle machines yourself**, including while collecting/choosing finalists. Reallocation of allowances or raising the total cap requires a researcher decision.

## 3. Screening launch

Checkpoint retention needs more disk than one live learner: conservative cumulative free-space gates are A 60 GB, B 70 GB, C 220 GB. These include a margin over the report's replay estimates; refine them with measured checkpoint sizes. Storage charges are outside the compute estimate. Moving completed checkpoints off-machine can reclaim space, but preserve them if you need resume evidence; collection bundles deliberately omit them.

Wrappers work from any cwd and require `--python`. B/C use `run_b_tuning.sh` / `run_c_tuning.sh`. Dry runs print exact commands without creating files.

```bash
bash run_scripts/tuning/run_a_tuning.sh --python "$PY" --dry-run
bash run_scripts/tuning/run_a_tuning.sh --python "$PY" --wandb-mode online --billing-start 2026-10-28T12:00:00Z
```

Use your actual billing timestamp, not the example. Logs and `status.csv` live under `results/v3/full/b/<role>/<batch>/`. A failed run makes the batch exit nonzero and print `BATCH FAILED`; other registered trials can still finish. Budget pauses also exit nonzero. Never run two launchers for the same role/root concurrently. Run directories are compact `v3/<mode>/<env>/<algo>/tNN/sNNN/aNN` with unique config hashes and W&B identities in project `new_experiments`, group `tuning-v3`.

## 4. Continue/restart

Repeat the same launch with `--continue-batch` and the same billing terms. It verifies complete runs before skipping them, resumes the identified trial's restart state if available, or creates a new attempt when no restart exists. Restarts are episode-boundary, not bitwise continuation. Corrupt restart files fail visibly; preserve the failed evidence and quarantine the corrupt checkpoint directory before retrying. Never substitute a different candidate's/baseline checkpoint. Duplicate successful attempts block selection; pass an explicit config-hash → relative-attempt JSON with `--attempt-choices` to the selection tools after reconciliation.

## 5. Collect, then freeze confirmation assignments

If duplicates required an explicit attempt choice, pass the same `--attempt-choices choices.json` to launch, manifest, collection and selection. The JSON maps a full config hash to its chosen relative `v3/.../aNN` directory; other attempts remain evidence. Collection manifests use unique content-derived filenames so independent machine bundles do not overwrite each other's manifests.

```bash
"$PY" -B run_scripts/tuning/collect.py --out tuning-a-screen.zip
```

Collect each machine's bundle, unpack into a merged tuning results root, and reconcile collisions without overwriting differing evidence. Bundles contain CSV/JSON only (including the exact trial list and collection manifest); no replay, policy or W&B cache. Keep checkpoints on their source machines for resumption. Optional finalist policies can be copied separately after checking size. Preserve failed attempts, identity and provenance. Paths are checked against the Windows checkout prefix and must remain below 240 characters.

```bash
"$PY" -B run_scripts/tuning/check_manifest.py --stage screen
"$PY" -B run_scripts/tuning/select_configs.py --prepare-confirmation
```

Missing/failed/invalid screening results prevent selection. An explicit `--allow-incomplete --reason "..."` records a deviation; it cannot fabricate metrics or waive the need for two valid candidates. Numerical failures stay unavailable. Resolve infrastructure failures by retrying identical settings. Copy the frozen `v3/full/finalists.json` to **all three machines**. It fixes candidate assignments to confirmation slots; do not regenerate it after inspecting confirmation outcomes.

## 6. Confirmation and final selection

Repeat each role's launch with `--stage confirm --continue-batch` and its billing terms. Confirmation uses seeds 102–104, fresh models and the same reduced budget. Collect again, merge evidence, then:

```bash
"$PY" -B run_scripts/tuning/check_manifest.py
"$PY" -B run_scripts/tuning/select_configs.py
```

Require all **154/154**, with A **27**, B **74**, C **53**. `selection_report.md` and `selected_configs.json` are written under `results/v3/full/`. Selection uses the mean confirmation AUC tuple, reports variability and noisy AUC, and preserves screening evidence. A missing confirmation never becomes zero and cannot be waived. An incomplete-data override makes outputs ineligible for an automatic final comparison until separately reviewed. No final-comparison or stress execution is implemented.

## 7. PowerShell collection check and local smoke

```powershell
$py = 'D:\miniconda3\envs\temporal_gym\python.exe'
$m = (& $py -B run_scripts/tuning/check_manifest.py --json | ConvertFrom-Json)
$m.counts
$m.by_role | Format-List
$m.entries | Group-Object role,state | Select-Object Name,Count
```

Local end-to-end diagnostic (sequential CPU, no login):

```powershell
& $py -B run_scripts/tuning/launch.py --role all --mode smoke --device-policy cpu --wandb-mode offline --continue-batch
& $py -B run_scripts/tuning/select_configs.py --mode smoke --prepare-confirmation
& $py -B run_scripts/tuning/launch.py --role all --stage confirm --mode smoke --device-policy cpu --wandb-mode offline --continue-batch
& $py -B run_scripts/tuning/select_configs.py --mode smoke
& $py -B run_scripts/tuning/check_manifest.py --mode smoke
```

## 8. Output contract and cloud gates

Each attempt retains the baseline guide §9 contract: resolved config, identity, metadata, status, resources, train episodes/updates, periodic episodes/points and terminal evaluation CSVs, final summary, plus v3 noisy counterparts. Identity adds candidate, stage, slot, plan hash and confirmation assignment hash. Paths point only to the trial's own records; no newest-folder discovery. Latest restart and final policy stay on-machine. Smoke selection explicitly has `usable_for_final_comparison=false`.

Cloud gates still require GPU/CUDA learning checks, online W&B connectivity and resume behaviour, Linux bash launch/interrupt tests, measured v3 cost and archive transfer between three machines. Local tests cannot establish these. Compare observed throughput to the planning estimates before proceeding; never silently reduce fidelity or drop cells.
