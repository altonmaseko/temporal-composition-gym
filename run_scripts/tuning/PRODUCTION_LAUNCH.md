# Tuning v3 production launch and collection

Run each block exactly as written, one at a time, on the named computer. Machine commands are Bash; laptop commands are PowerShell. The researcher starts the instances and supplies authentication. No script rents or destroys a machine.

## 1. Shared revision

Validated code commit: `af2d21a58bc01247f88cd5f74a853628563b923e`, branch `new_experiments`. Push the implementation and this guide yourself before cloning. Later documentation/finalist/results commits are permitted if the code check in A passes. Each run records its revision and working-tree metadata.

## 2. What will run

Entries are screening/confirmation run counts. Candidates t00–t07 train on seed **101**; top two train fresh on **102,103,104**. The assignment to confirmation slots is frozen in finalists.json.

| Cell | A | B | C |
|---|---:|---:|---:|
| Delivery DQN | 0/0 | 8/6 | 0/0 |
| Delivery DQN-TS | 0/0 | 8/6 | 0/0 |
| Delivery DRQN | 6/0 | 2/6 | 0/0 |
| Delivery QRM | 0/0 | 8/6 | 0/0 |
| Colour DQN | 0/0 | 0/0 | 8/6 |
| Colour DQN-TS | 0/0 | 0/0 | 8/6 |
| Colour DRQN | 7/0 | 1/6 | 0/0 |
| Colour QRM-Cue | 0/0 | 0/0 | 8/6 |
| Maintenance PPO | 0/0 | 8/6 | 0/0 |
| Maintenance Recurrent PPO | 8/6 | 0/0 | 0/0 |
| Maintenance PPO-RM | 0/0 | 3/0 | 5/6 |
| Stages | **21/6** | **38/36** | **29/24** |
| Total | **27** | **74** | **53** |

Total **88 screening +66 confirmation =154**. Per-run budget: delivery 250,000; colour 750,000; maintenance 512,000 interactions (25% fidelity; late learners can be missed). Evaluate step 0 and every 5% (12,500/37,500/25,600), with 50/50/20 episodes at **3,000,000+i**. Terminal validation: 200/200/100 episodes at **3,010,000+i**. Greedy success AUC, then task progress AUC, then raw return AUC, then candidate ID; confirmation uses the mean over three fresh seeds. Report noisy evaluation without optimizing it (D5). W&B project **new_experiments**, group **tuning-v3**; smoke/preflight groups **tuning-v3-smoke / tuning-v3-preflight**.

| Role | Linear reference hours | Planning billed hours | At $0.30/h | Hard share |
|---|---:|---:|---:|---:|
| A | 10.67 | 21–32 | $6.30–9.60 | 33.34 h |
| B | 10.64 | 21–32 | $6.30–9.60 | 33.33 h |
| C | 10.61 | 21–32 | $6.30–9.60 | 33.33 h |

The 2–3× allowance covers extra evaluation overhead; these are estimates, not measured v3 throughput. **100 billed hours / ~$30 compute total**, including setup/idle/retries. Storage and transfer are extra. Pause before exceeding a share; do not silently reduce fidelity or remove cells.

## 3. Three machines and billing

| Role | Instance | Connect from laptop | GPU | RAM | Rent disk | Required free after install |
|---|---|---|---|---:|---:|---:|
| A | <INSTANCE> | ssh -p <PORT> root@<IP> | RTX3090 24GB | ≥16GB | ≥40GB | ≥20GB |
| B | <INSTANCE> | ssh -p <PORT> root@<IP> | RTX3090 24GB | ≥16GB | ≥40GB | ≥20GB |
| C | <INSTANCE> | ssh -p <PORT> root@<IP> | RTX3090 24GB | ≥16GB | ≥50GB | ≥25GB |

Choose offers at **$0.30/h or less** and account conservatively at 0.30 in these commands. If unavailable, revise the cost plan before launching. One process at a time per machine. Provisioned disk includes OS/environment space; preflight checks actual free space.

Disk estimates allow 100MB retained final policy per run (2.7/7.4/5.3GB), live replay plus transactional replacement (up to 4.6/4.6/8.4GB), CSV/JSON, SDK cache and margin. Measure policy/replay peaks on the first completed runs. Production defaults to **--prune-completed-restarts**: delete restart, restart.old and restart.tmp only after independent verification of that attempt. Keep final policies and all evidence. Unverified/failed/ambiguous/unfinished runs are never pruned. To opt out, add **--no-prune-completed-restarts** to both launcher and preflight and provision **60/70/220GB free** instead.

An SSH login may already attach tmux. The A block attaches/creates a named session; run subsequent blocks inside it. Disconnecting SSH does not stop training. Reconnect with the same SSH command, then reattach tmux if necessary.

The billing start is the **UTC time you start the instance in the Vast.ai console**, including setup time. Replace only **<BILLING_START>** with that timestamp, e.g. 2026-10-29T08:15:00Z. The helper stores it in ~/tuning_a.env (or b/c). Launch passes **--billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS"**. First session previous hours =0 automatically. Never replace the start time on a retry. Section 7 carries earlier sessions forward when stopping/restarting.

## 4. Machine A: SCREENING

Connect from laptop PowerShell with this machine's port/IP:
```powershell
ssh -p <PORT> root@<IP>
```

**A. Setup**, once. Expected: attached tmux, then successful clone:
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_a; fi
```
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```
Code check: expected commit line, then **CODE CHECK DONE**, no diff. A nonzero exit means stop:
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --exit-code af2d21a58bc01247f88cd5f74a853628563b923e HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && git diff --exit-code HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && echo "CODE CHECK DONE"
```
Create venv and install. Expected: Python version, successful pip exit; resolve installation errors before B:
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in yourself; expected W&B login success:
```bash
~/tcg/bin/wandb login
```
Store actual console rental start. Expected **Session ready: 0.000000 previous hours, rate $0.30/h**:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role a --start '<BILLING_START>'
```

**B. Preflight.** Must end **PREFLIGHT PASSED**; resolve every FAIL before C:
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/preflight.py --role a --device-policy plan --wandb-mode online --learning-check 2>&1 | tee ~/preflight_a.log
```

**C. Launch**, 21 screening runs. Expected sequential START/COMPLETE, final **BATCH COMPLETE**, exit 0:
```bash
set -o pipefail
source ~/tuning_a.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_a_tuning.sh --python ~/tcg/bin/python --stage screen --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_a.log
```

**D. Wait.** Reconnect if needed:
```bash
tail -n 20 ~/tuning_a.log
```
Finished when this invocation ends with exactly **BATCH COMPLETE**. If BATCH FAILED, follow section 9. Expected **21/21 complete**:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --stage screen --role a
```

**E. Bundle/fingerprint.** Expected file count, **no checkpoints**, then SHA256 (also saved to sidecar). Save the printed hash:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_a_screen.zip && sha256sum ~/tuning_a_screen.zip | tee ~/tuning_a_screen.zip.sha256
```
Collector refuses an existing archive: preserve earlier evidence before making a replacement.

## 5. Machine B: SCREENING

Connect from laptop PowerShell with this machine's port/IP:
```powershell
ssh -p <PORT> root@<IP>
```

**A. Setup**, once. Expected: attached tmux, then successful clone:
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_b; fi
```
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```
Code check: expected commit line, then **CODE CHECK DONE**, no diff. A nonzero exit means stop:
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --exit-code af2d21a58bc01247f88cd5f74a853628563b923e HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && git diff --exit-code HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && echo "CODE CHECK DONE"
```
Create venv and install. Expected: Python version, successful pip exit; resolve installation errors before B:
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in yourself; expected W&B login success:
```bash
~/tcg/bin/wandb login
```
Store actual console rental start. Expected **Session ready: 0.000000 previous hours, rate $0.30/h**:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role b --start '<BILLING_START>'
```

**B. Preflight.** Must end **PREFLIGHT PASSED**; resolve every FAIL before C:
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/preflight.py --role b --device-policy plan --wandb-mode online --learning-check 2>&1 | tee ~/preflight_b.log
```

**C. Launch**, 38 screening runs. Expected sequential START/COMPLETE, final **BATCH COMPLETE**, exit 0:
```bash
set -o pipefail
source ~/tuning_b.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_b_tuning.sh --python ~/tcg/bin/python --stage screen --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_b.log
```

**D. Wait.** Reconnect if needed:
```bash
tail -n 20 ~/tuning_b.log
```
Finished when this invocation ends with exactly **BATCH COMPLETE**. If BATCH FAILED, follow section 9. Expected **38/38 complete**:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --stage screen --role b
```

**E. Bundle/fingerprint.** Expected file count, **no checkpoints**, then SHA256 (also saved to sidecar). Save the printed hash:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_b_screen.zip && sha256sum ~/tuning_b_screen.zip | tee ~/tuning_b_screen.zip.sha256
```
Collector refuses an existing archive: preserve earlier evidence before making a replacement.

## 6. Machine C: SCREENING

Connect from laptop PowerShell with this machine's port/IP:
```powershell
ssh -p <PORT> root@<IP>
```

**A. Setup**, once. Expected: attached tmux, then successful clone:
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_c; fi
```
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```
Code check: expected commit line, then **CODE CHECK DONE**, no diff. A nonzero exit means stop:
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --exit-code af2d21a58bc01247f88cd5f74a853628563b923e HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && git diff --exit-code HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && echo "CODE CHECK DONE"
```
Create venv and install. Expected: Python version, successful pip exit; resolve installation errors before B:
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in yourself; expected W&B login success:
```bash
~/tcg/bin/wandb login
```
Store actual console rental start. Expected **Session ready: 0.000000 previous hours, rate $0.30/h**:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role c --start '<BILLING_START>'
```

**B. Preflight.** Must end **PREFLIGHT PASSED**; resolve every FAIL before C:
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/preflight.py --role c --device-policy plan --wandb-mode online --learning-check 2>&1 | tee ~/preflight_c.log
```

**C. Launch**, 29 screening runs. Expected sequential START/COMPLETE, final **BATCH COMPLETE**, exit 0:
```bash
set -o pipefail
source ~/tuning_c.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_c_tuning.sh --python ~/tcg/bin/python --stage screen --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_c.log
```

**D. Wait.** Reconnect if needed:
```bash
tail -n 20 ~/tuning_c.log
```
Finished when this invocation ends with exactly **BATCH COMPLETE**. If BATCH FAILED, follow section 9. Expected **29/29 complete**:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --stage screen --role c
```

**E. Bundle/fingerprint.** Expected file count, **no checkpoints**, then SHA256 (also saved to sidecar). Save the printed hash:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_c_screen.zip && sha256sum ~/tuning_c_screen.zip | tee ~/tuning_c_screen.zip.sha256
```
Collector refuses an existing archive: preserve earlier evidence before making a replacement.

## 7. Between stages: laptop PowerShell and machine handoff

Keep machines running only for a handoff of a few minutes within the cap. Otherwise **stop them in Vast.ai with disk kept**, so only storage is billed. Do not destroy after screening: confirmation and policies still need those disks.

Before stopping each finished machine, run its close block **once**. Expected: carried hours and an instruction to stop within one hour. The helper adds a conservative **one-hour** allowance for the interval until the console stop, and archives the current ledger. Stop in the console within that hour; if missed, correct accounting upward before continuing.

Machine A before stopping:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role a --close
```

Machine B before stopping:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role b --close
```

Machine C before stopping:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role c --close
```

If keeping a machine running, omit close/start and keep its original env values. If stopped, restart the **same instance/disk** in the console, reconnect (port/IP can change), and reattach tmux. Use the new console start time below. Expected: Session ready with **nonzero** previous hours, including all earlier sessions plus stop allowance. No billing ledger needs to be edited or deleted manually.

Machine A after restart:
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_a; fi
```
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role a --start '<BILLING_START>'
```

Machine B after restart:
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_b; fi
```
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role b --start '<BILLING_START>'
```

Machine C after restart:
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_c; fi
```
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role c --start '<BILLING_START>'
```

On laptop PowerShell initialize these variables; retain this window through section 10:
```powershell
$ErrorActionPreference = 'Stop'
$Repo = 'C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym'
$Python = 'D:\miniconda3\envs\temporal_gym\python.exe'
$ResultsRoot = Join-Path $Repo 'run_scripts\tuning\results'
$Mode = 'full'
$Bundles = Join-Path $HOME 'Downloads'
Set-Location -LiteralPath $Repo
```
Download each machine's archive and hash sidecar. Expected exit 0; Get-FileHash must equal both the machine's printed sha256sum and sidecar:
```powershell
scp -P <PORT> root@<IP>:tuning_a_screen.zip root@<IP>:tuning_a_screen.zip.sha256 $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Screening download failed' }
(Get-FileHash (Join-Path $Bundles 'tuning_a_screen.zip')).Hash.ToLower()
Get-Content (Join-Path $Bundles 'tuning_a_screen.zip.sha256')
```
```powershell
scp -P <PORT> root@<IP>:tuning_b_screen.zip root@<IP>:tuning_b_screen.zip.sha256 $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Screening download failed' }
(Get-FileHash (Join-Path $Bundles 'tuning_b_screen.zip')).Hash.ToLower()
Get-Content (Join-Path $Bundles 'tuning_b_screen.zip.sha256')
```
```powershell
scp -P <PORT> root@<IP>:tuning_c_screen.zip root@<IP>:tuning_c_screen.zip.sha256 $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Screening download failed' }
(Get-FileHash (Join-Path $Bundles 'tuning_c_screen.zip')).Hash.ToLower()
Get-Content (Join-Path $Bundles 'tuning_c_screen.zip.sha256')
```

Merge, verifying hashes again and rejecting differing existing files. Expected **BUNDLES VERIFIED AND MERGED**:
<!-- rehearsal:screen-merge -->
```powershell
& "$Repo\run_scripts\tuning\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @((Join-Path $Bundles 'tuning_a_screen.zip'), (Join-Path $Bundles 'tuning_b_screen.zip'), (Join-Path $Bundles 'tuning_c_screen.zip'))
```
Expected **88/88 complete**, A21/B38/C29 screening; whole-study still has 66 pending confirmation slots. Freeze then fingerprint (expected **Finalists frozen**, followed by a 64-character hash):
<!-- rehearsal:screen-check -->
```powershell
& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --results-root $ResultsRoot --mode $Mode --stage screen
if ($LASTEXITCODE -ne 0) { throw 'Screening incomplete' }
& $Python -B "$Repo\run_scripts\tuning\select_configs.py" --results-root $ResultsRoot --mode $Mode --prepare-confirmation
if ($LASTEXITCODE -ne 0) { throw 'Finalist freeze failed' }
$Finalists = Join-Path $ResultsRoot "v3\$Mode\finalists.json"
$FinalistHash = (Get-FileHash $Finalists).Hash.ToLower()
$FinalistHash
```
Save that hash. Commit only the assignment now; reset staging first without changing working files. Expected staged filename **only finalists.json**:
```powershell
git restore --staged .
git add -- run_scripts/tuning/results/v3/full/finalists.json
$files = @(git diff --cached --name-only)
if ($files.Count -ne 1 -or $files[0] -ne 'run_scripts/tuning/results/v3/full/finalists.json') { throw 'Unexpected staged files' }
git commit -m "Freeze v3 tuning finalist assignments from screening"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin new_experiments
if ($LASTEXITCODE -ne 0) { throw 'Finalist push failed' }
```
On each machine after that push, whether continuously running or restarted, run this exact block. Expected fast-forward and SHA256 equal to laptop $FinalistHash:
```bash
cd ~/temporal-composition-gym && git pull --ff-only origin new_experiments && sha256sum run_scripts/tuning/results/v3/full/finalists.json
```
If an untracked finalists file blocks pull, compare it with the frozen hash and reconcile; do not force-overwrite. Do not copy another machine's screening records onto a machine: its bundle/counts should contain its own assigned runs.

## 8. CONFIRMATION per machine

All three finalist hashes must match before launch. Fresh seeds 102–104 use the same reduced budget.

**Machine A**, 6 confirmation runs:
```bash
set -o pipefail
source ~/tuning_a.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_a_tuning.sh --python ~/tcg/bin/python --stage confirm --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_a.log
```
Wait:
```bash
tail -n 20 ~/tuning_a.log
```
Expected final **BATCH COMPLETE**. Bundle only after this role's full count passes; expected **27/27 complete**, file count and fingerprint:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --role a && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_a_confirm.zip && sha256sum ~/tuning_a_confirm.zip | tee ~/tuning_a_confirm.zip.sha256
```

**Machine B**, 36 confirmation runs:
```bash
set -o pipefail
source ~/tuning_b.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_b_tuning.sh --python ~/tcg/bin/python --stage confirm --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_b.log
```
Wait:
```bash
tail -n 20 ~/tuning_b.log
```
Expected final **BATCH COMPLETE**. Bundle only after this role's full count passes; expected **74/74 complete**, file count and fingerprint:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --role b && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_b_confirm.zip && sha256sum ~/tuning_b_confirm.zip | tee ~/tuning_b_confirm.zip.sha256
```

**Machine C**, 24 confirmation runs:
```bash
set -o pipefail
source ~/tuning_c.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_c_tuning.sh --python ~/tcg/bin/python --stage confirm --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_c.log
```
Wait:
```bash
tail -n 20 ~/tuning_c.log
```
Expected final **BATCH COMPLETE**. Bundle only after this role's full count passes; expected **53/53 complete**, file count and fingerprint:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --role c && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_c_confirm.zip && sha256sum ~/tuning_c_confirm.zip | tee ~/tuning_c_confirm.zip.sha256
```

## 9. Monitoring, failures and restart

Read ~/tuning_a.log, ~/tuning_b.log or ~/tuning_c.log. Per-trial stdout/stderr are under results/v3/full/b/role/batch/NN.out.log and NN.err.log; status.csv has trial/state/exit_code/seconds. A failed run's status.json has the error and traceback.

After interruption or **BATCH FAILED**, inspect that error. **Never repeat a plain launch**: use the appropriate restart below, with the same billing env. It skips verified complete trials, resumes an unfinished trial's own checkpoint or starts a new attempt if no restart exists. A repeat failure needs investigation; preserve all evidence. Resume is episode-boundary, not bitwise continuation.

Machine A, screen restart:
```bash
set -o pipefail
source ~/tuning_a.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_a_tuning.sh --python ~/tcg/bin/python --stage screen --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_a.log
```

Machine A, confirm restart:
```bash
set -o pipefail
source ~/tuning_a.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_a_tuning.sh --python ~/tcg/bin/python --stage confirm --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_a.log
```

Machine B, screen restart:
```bash
set -o pipefail
source ~/tuning_b.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_b_tuning.sh --python ~/tcg/bin/python --stage screen --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_b.log
```

Machine B, confirm restart:
```bash
set -o pipefail
source ~/tuning_b.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_b_tuning.sh --python ~/tcg/bin/python --stage confirm --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_b.log
```

Machine C, screen restart:
```bash
set -o pipefail
source ~/tuning_c.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_c_tuning.sh --python ~/tcg/bin/python --stage screen --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_c.log
```

Machine C, confirm restart:
```bash
set -o pipefail
source ~/tuning_c.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_c_tuning.sh --python ~/tcg/bin/python --stage confirm --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_c.log
```

**Budget pause:** stop work and stop the instance in the console before its allowance expires. The launcher stops a child five minutes before its deadline; it cannot stop billing. Do not reset billing terms to bypass the cap.

**Duplicates:** inspect both attempts' identity/failure/resume evidence, decide the intended infrastructure attempt before comparing scores, and preserve both. Save a JSON object mapping full config hash to relative v3/full/.../aNN path as run_scripts/tuning/results/v3/full/choices.json. Add **--attempt-choices run_scripts/tuning/results/v3/full/choices.json** to every matching launcher/manifest/collect/select command. Only a verified complete attempt can be chosen; the JSON output records that explicit choice. Keep this same choices file in publication. Missing or failed screening blocks freeze; a deliberate incomplete-data deviation needs researcher review and --allow-incomplete --reason. Six valid confirmations remain mandatory.

**Replacement archive:** keep the old archive/sidecar under timestamped names before repeating the bundle command. Do not overwrite differing laptop evidence. The merge tool stops on conflicts; preserve both versions and reconcile the affected attempts. Confirmation archives normally add new records while retaining identical screening records. Live accounting snapshots are content-addressed to avoid conflicts across rental sessions.

If a replacement is necessary, run this on that machine before repeating its bundle block. Expected: exit 0; existing archives and sidecars are preserved with the same UTC suffix (nothing is deleted):

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
for name in tuning_a_screen.zip tuning_a_confirm.zip tuning_b_screen.zip tuning_b_confirm.zip tuning_c_screen.zip tuning_c_confirm.zip; do
    if [ -f "$HOME/$name" ]; then mv -n "$HOME/$name" "$HOME/$name.$stamp"; fi
    if [ -f "$HOME/$name.sha256" ]; then mv -n "$HOME/$name.sha256" "$HOME/$name.sha256.$stamp"; fi
done
```

## 10. Final collect and publish on laptop

Use the section 7 PowerShell variables. Download, print and compare each hash with its machine:
```powershell
scp -P <PORT> root@<IP>:tuning_a_confirm.zip root@<IP>:tuning_a_confirm.zip.sha256 $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Confirmation download failed' }
(Get-FileHash (Join-Path $Bundles 'tuning_a_confirm.zip')).Hash.ToLower()
Get-Content (Join-Path $Bundles 'tuning_a_confirm.zip.sha256')
```
```powershell
scp -P <PORT> root@<IP>:tuning_b_confirm.zip root@<IP>:tuning_b_confirm.zip.sha256 $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Confirmation download failed' }
(Get-FileHash (Join-Path $Bundles 'tuning_b_confirm.zip')).Hash.ToLower()
Get-Content (Join-Path $Bundles 'tuning_b_confirm.zip.sha256')
```
```powershell
scp -P <PORT> root@<IP>:tuning_c_confirm.zip root@<IP>:tuning_c_confirm.zip.sha256 $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Confirmation download failed' }
(Get-FileHash (Join-Path $Bundles 'tuning_c_confirm.zip')).Hash.ToLower()
Get-Content (Join-Path $Bundles 'tuning_c_confirm.zip.sha256')
```

<!-- rehearsal:final-merge -->
```powershell
& "$Repo\run_scripts\tuning\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @((Join-Path $Bundles 'tuning_a_confirm.zip'), (Join-Path $Bundles 'tuning_b_confirm.zip'), (Join-Path $Bundles 'tuning_c_confirm.zip'))
```
Expected **BUNDLES VERIFIED AND MERGED**. Check and select:
<!-- rehearsal:final-check -->
```powershell
& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --results-root $ResultsRoot --mode $Mode
if ($LASTEXITCODE -ne 0) { throw 'Final manifest incomplete' }
$m = (& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --results-root $ResultsRoot --mode $Mode --json | ConvertFrom-Json)
$m.by_role | Format-List
& $Python -B "$Repo\run_scripts\tuning\select_configs.py" --results-root $ResultsRoot --mode $Mode
if ($LASTEXITCODE -ne 0) { throw 'Final selection failed' }
```
Expected **154/154 complete**, A27/B74/C53, then **{"complete": true, "problems": []}**. Back up all six archives and sidecars outside the laptop before destroying machines. Each machine can be destroyed only after its confirmation bundle hash matches, backup exists and its combined counts pass:

Expected 27/27:
```powershell
& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --role a
if ($LASTEXITCODE -ne 0) { throw 'Role a incomplete; retain instance' }
```

Expected 74/74:
```powershell
& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --role b
if ($LASTEXITCODE -ne 0) { throw 'Role b incomplete; retain instance' }
```

Expected 53/53:
```powershell
& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --role c
if ($LASTEXITCODE -ne 0) { throw 'Role c incomplete; retain instance' }
```

Policies remain on instances; copy desired finalist policies separately before destruction. Inspect selected config provenance and report, then stage only CSV/JSON and the report:
```powershell
Set-Location -LiteralPath $Repo
git restore --staged .
@("$ResultsRoot\v3\full", "$ResultsRoot\collection", "$ResultsRoot\accounting") | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File } | Where-Object { $_.Extension -in '.csv','.json' -and $_.FullName -notmatch '[\\/]checkpoints[\\/]' } | ForEach-Object { git add -- $_.FullName }
git add -- run_scripts/tuning/results/v3/full/selection_report.md
$staged = @(git diff --cached --name-only)
if ($staged | Where-Object { $_ -match 'checkpoints|\.pyc$|_wb/' }) { throw 'Checkpoint/cache staged' }
git diff --cached --stat
```
Expected only tuning CSV/JSON/report, including finalists.json and selected_configs.json. No checkpoints/bytecode:
```powershell
git commit -m "Publish verified v3 tuning results and selected configurations"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin new_experiments
if ($LASTEXITCODE -ne 0) { throw 'Results push failed' }
```

| Artifact | Laptop | GitHub | W&B |
|---|---|---|---|
| CSV/JSON, trial list, finalists, selected configs, report | yes | after publication | configs/metrics separately |
| Archives/sidecars | Downloads + external backup | no | no |
| Greedy/noisy curves and summaries | CSV/JSON | CSV/JSON | tuning-v3 |
| Final policies | if separately copied | no | no automatic upload |
| Completed restart/replay | pruned at source | no | no |
| Unfinished restart/replay | source until resolved | no | no |
| Per-run stdout/stderr and SDK cache | source machine | no | SDK sync only |

## 11. Local validation and open cloud checks

T2: **107 tests**, **88/88** CPU/offline diagnostic runs (22 screening +66 confirmation), 11 selections without override; all selected success AUCs zero. Four Windows checkpoint failures recovered; one failed attempt and three pre-resume statuses retained. Injected failure resumed from step32. All **440 baseline hashes** unchanged; **55/55** baseline manifest. Original bundle: **1,286** CSV/JSON members, **735,954 bytes**, longest extracted path **223** characters. Thirty-one early metadata files lack Git revision; this is engineering smoke evidence, not production science. See LOCAL_VALIDATION.md.

T3 results are in T3_VALIDATION.md. Locally rehearse the marked PowerShell merge/freeze/distribution/selection commands with temporary smoke roots. Authentication, network transfers and pushes are not executed in rehearsal. Cloud-only: requirements installation, CUDA/VRAM/real GPU updates, online W&B/login/resume, native Linux interruption/continuation, SSH/scp, measured v3 throughput/disk peaks and billed cost. Each cloud preflight must pass before launch.

To reproduce the offline rehearsal on the laptop, run this exact PowerShell block. Expected **REHEARSAL PASSED**, with 22/22 screening, 88/88 final and three identical finalist hashes. It executes the marked guide blocks verbatim; local file copy/hash checks stand in for GitHub transport and the three machine pulls. It creates an isolated short temporary directory under results and leaves production results untouched:

```powershell
& "$Repo\run_scripts\tuning\rehearse.ps1"
```

## 12. Collection contract for the stress-test phase

Root: run_scripts/tuning/results/v3/full. Attempts: env/algo/tNN/sNNN/aNN. All extracted/generated Windows paths must be below240 characters. Downstream stress execution is separate.

| File | Fields and meaning |
|---|---|
| trials_v3.json | schema, protocol_version, phase, sampler/seed, screen/confirmation seeds, cells/spaces/candidates (hyperparams, full config, config_hash), slot_roles, plan_hash |
| finalists.json | schema, plan_hash, mode, full screening cells/evidence/unresolved states, two finalist IDs/cell, allow_incomplete/reason, assignment_hash; frozen before confirmation |
| selected_configs.json | schema, phase, protocol_version, mode, plan_hash, allow_incomplete, override_reason, problems, complete, usable_for_final_comparison; selected per cell: candidate, hyperparams, objective mean tuple, std, noisy_mean, per-seed evidence, screening, both finalists, tie_break, zero_success, config_hashes; global screening/confirmation |
| selection_report.md | completion/deviations, greedy/noisy AUCs, seed SD and tie-break reasons |
| config.resolved.json | env/task kwargs, algo/hyperparams, seed/reset, budget, eval/cadence/seed blocks/secondary policy, checkpoint/logging/W&B config, tuning candidate/learner hash |
| identity.json | run_id, run_uid, wandb_id, config_hash, protocol_version, phase, mode, env, algo, family, seed, machine_role, batch_id, results_root, run_dir, created_utc, plan_hash, candidate, stage, slot, assignment_hash |
| metadata.json / metadata_resume_*.json | revision/branch/dirty state, Python/packages, hardware/device/threads, safe env vars, argv, launcher paths |
| status.json / status_resume_*.json | state/timestamps, budget/env_steps_done, verified, final_eval_done, n_updates, resumes/restored_steps/new_steps_this_session, W&B id/mode/url, error/traceback when failed |
| final_summary.json | global_step, n_updates, train_episodes, final_eval, final_eval_noisy, last_periodic_eval, config_hash, protocol_version, run_id, completed_utc |
| eval_points.csv and eval_points_noisy.csv | eval_index, global_step, n_episodes, return_mean/std/min/max, success_rate, length_mean, progress_mean, causes_json, n_updates |
| eval_periodic.csv and eval_periodic_noisy.csv | eval_index, global_step, episode_seed, return_raw, length, success, cause, progress |
| eval_final.csv and eval_final_noisy.csv | global_step, episode_seed, return_raw, length, success, cause, progress |
| train_episodes.csv | global_step, episode, return_raw, return_rm, length, success, cause, progress, stage_exits, wall_touches |
| train_updates.csv | global_step, n_updates, epsilon, loss, q_mean, grad_norm, extra_json |
| resources.csv | global_step, wall_time_s, env_steps_per_s, rss_gb, gpu_mem_alloc_gb, gpu_mem_max_gb, replay_frames |
| b/role/batch/status.csv | trial, state, exit_code, seconds; stdout/stderr remain at source |
| collection/*.json, accounting/*.json, archived bill sessions | content-addressed collected manifests and billing snapshots |
| choices.json / resume CSV/JSON backups, when present | explicit duplicate decisions and preserved pre-resume evidence |

Success/progress are task-defined (deliveries/correct walls/services), return is raw environment reward. Headline is greedy, noise remains secondary. Every objective comes from the identified attempt's own files, never a newest log folder. **Missing, failed, interrupted, invalid and nonfinite results are unavailable, never zero.** Verify exact budget, updates, evaluation records, seed independence and hashes before use. Policies and replay are outside CSV/JSON bundles; stress/final comparison must use its separately reserved seeds and protocol.
