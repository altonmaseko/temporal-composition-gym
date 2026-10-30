# Tuning v3: production launch and collection

## How to use this guide

- **Bash** blocks run on a rented machine, in its SSH window. **PowerShell** blocks run on your laptop.
- Copy each block exactly. Run one block at a time. After each block, check the **You should see** line before going on.
- No script rents, stops or destroys a machine. You do that yourself on the Vast.ai website.
- In tmux, open a spare window with **Ctrl+b** then **c**, and go back with **Ctrl+b** then **0**. **Never press Ctrl+C in the training window**: it interrupts the run (if it happens, see Part 9).

## The order of work

| Step | Where | What | Part |
|---|---|---|---|
| 1 | Each machine | Set up, preflight, start screening | 1 |
| 2 | Each machine | Wait; check now and then | 2 |
| 3 | Each screening machine, as soon as **it** prints **`BATCH COMPLETE`** | Pack, download, back up, destroy the screening machine | 3 |
| 4 | Laptop, once **all three** screening bundles are saved | Merge results, freeze finalists, push | 4 |
| 5 | Three newly rented machines | Send the three new SSH commands and IDs, then set up and start confirmation | 5 |
| 6 | Each confirmation machine, as soon as **it** prints **`BATCH COMPLETE`** | Download and back up results **and models**, then **destroy that machine immediately** | 6 |
| 7 | Laptop, once **all three** are downloaded | Final merge, selection, publish | 7 |
| 8 | Vast.ai | Check that the instances already destroyed in Parts 3 and 6 are gone | 8 |

The machines do not finish at the same time. Do Part 3 for each screening machine as soon as it finishes. Confirmation uses three new rentals; the old screening machines are not restarted. **For confirmation, finish that machine's Part 6 steps and destroy it immediately, even while the other two are running. Only Part 7 waits for all three downloads.**

Expected running time per stage (estimated 2026-10-29 from the first measured runs; colour DRQN and Recurrent PPO were not yet measured, so A and B are the least certain):

| Machine | Screening (Parts 1–2) | Confirmation (Part 5) |
|---|---:|---:|
| A | ~15–18 h | ~2.5–3 h |
| B | ~5–7 h | ~12–15 h |
| C | ~10–12 h | ~8–9 h |

Screening ends when the slowest machine (A) finishes. Confirmation ends when the slowest machine (B) finishes. The confirmation estimates above are unchanged; fresh installation and preflight take additional time, and different hardware can change actual speed. Training steps, seeds and evaluation settings are unchanged.

## Part 1. Set up and start screening (done 2026-10-29)

Launch code commit: `fe837314e40662649e0fe1bbfb7d88cfe2307d5b`, branch `new_experiments`. Push the code before cloning. Later commits that only change documentation or results are fine; the code check below makes sure the code itself has not changed.

The original **screening** machines (do not use these SSH addresses for confirmation):

| Role | Instance | Connect from laptop |
|---|---|---|
| A | 58230725 | `ssh -p 20136 root@93.241.214.122 -L 8080:localhost:8080` |
| B | 58230761 | `ssh -p 55203 root@108.240.82.27 -L 8080:localhost:8080` |
| C | 58230733 | `ssh -p 40318 root@159.48.242.36 -L 8080:localhost:8080` |

If you are connected to more than one machine at once, the second and third SSH windows warn that port 8080 is in use. That is harmless.

Do steps 1.1 to 1.7 on each machine. Below, `X` is the machine letter: replace `a` in every command with `b` on machine B and `c` on machine C.

**1.1 Connect** from laptop PowerShell, using the table above.

**1.2 Start tmux.** You should see: a tmux status bar.
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_a; fi
```

**1.3 Download the code.** You should see: the clone finish without errors.
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```

**1.4 Check the code.** You should see: one commit line, then **`CODE CHECK DONE`**. Anything else means stop.
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --exit-code fe837314e40662649e0fe1bbfb7d88cfe2307d5b HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && git diff --exit-code HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && echo "CODE CHECK DONE"
```

**1.5 Install Python packages.** You should see: a Python version, then pip finishing without errors.
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in to W&B yourself. You should see: a login success message.
```bash
~/tcg/bin/wandb login
```

**1.6 Record when billing started**: the UTC time you started the instance on Vast.ai. You should see: **`Session ready: 0.000000 previous hours, rate $0.30/h`**.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role a --start '2026-10-29T09:55:45Z'
```

**1.7 Preflight, then start screening.** The preflight must end with **`PREFLIGHT PASSED`**. Fix every FAIL before starting.
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/preflight.py --role a --device-policy plan --wandb-mode online --learning-check 2>&1 | tee ~/preflight_a.log
```
Start screening. You should see: `START ...` and `COMPLETE ...` lines, one run at a time.
```bash
set -o pipefail
source ~/tuning_a.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_a_tuning.sh --python ~/tcg/bin/python --stage screen --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_a.log
```

## Part 2. While screening runs

You can close SSH and turn off your laptop. Training keeps running.

To check on a machine, connect, open a spare tmux window (**Ctrl+b** then **c**) and run the line for that machine:

| Machine | Command |
|---|---|
| A | `tail -n 3 ~/tuning_a.log` |
| B | `tail -n 3 ~/tuning_b.log` |
| C | `tail -n 3 ~/tuning_c.log` |

What the last line means:

- `START ...`: still running. Leave it.
- **`BATCH COMPLETE`**: this machine is finished. Go to Part 3 for this machine.
- **`BATCH FAILED`** or **`PAUSED`**: something stopped or failed. Go to Part 9.
- **`BUDGET WARNING ...`**: the machine may go over its hour budget. It keeps running anyway. You decide whether to keep paying.

To see how long each finished run took (column `seconds`):
```bash
cat ~/temporal-composition-gym/run_scripts/tuning/results/v3/full/b/a/*/status.csv
```
Use `/b/b/` on machine B and `/b/c/` on machine C.

## Part 3. When a machine prints **`BATCH COMPLETE`** (screening)

Do these five steps for that machine right away. Don't wait for the other machines. **Download, verify and back up before destroying it.** Confirmation starts from fresh runs on new machines and needs the screening CSV/JSON evidence and frozen finalists, not the old checkpoints.

Destroying the machine also deletes its trained policies, raw stdout/stderr and W&B cache; these are not included in the results zip. Copy any of those files you want to retain while the machine is still running.

**Your current screening status:** you have completed Part 3 for **A (58230725)** and **B (58230761)** and destroyed both. **Skip all of 3A and 3B below; do not repeat their commands or rent replacements for screening.** Keep their downloaded bundles and backups for Part 4. B's verified bundle SHA-256 is **`c18f535e574650be9a306b33516257b6449a7dec843c3608aae415cf68b0c717`**. The full A/B steps remain below for reference. Complete Part 3C when screening machine C finishes.

### Machine A

**3A.1 On machine A: check and pack.** You should see: **`21/21 complete`**, then a file count with **no checkpoints**, then a long hash.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --stage screen --role a && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_a_screen.zip && sha256sum ~/tuning_a_screen.zip | tee ~/tuning_a_screen.zip.sha256
```
If it says less than 21/21, do not continue. Go to Part 9.

**3A.2 On laptop PowerShell: download.** You should see: **`HASH OK`**. If scp asks to trust the host, type `yes`.
```powershell
$Bundles = Join-Path $HOME 'Downloads'
scp -P 20136 root@93.241.214.122:tuning_a_screen.zip root@93.241.214.122:tuning_a_screen.zip.sha256 $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Download failed: do not stop the machine' }
$got = (Get-FileHash (Join-Path $Bundles 'tuning_a_screen.zip')).Hash.ToLower()
$want = (Get-Content (Join-Path $Bundles 'tuning_a_screen.zip.sha256')).Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not stop the machine' } else { "HASH OK $got" }
```

**3A.3 On machine A: close billing.** You should see: **`Stop the instance in the console within one hour`**.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role a --close
```

**3A.4 On laptop: back up and recheck.** Copy `tuning_a_screen.zip` and `tuning_a_screen.zip.sha256` from Downloads to a USB drive or cloud storage. Confirm the backup has finished. Recheck the Downloads copy; you should see **`HASH OK`**:
```powershell
$Bundles = Join-Path $HOME 'Downloads'
$got = (Get-FileHash (Join-Path $Bundles 'tuning_a_screen.zip')).Hash.ToLower()
$want = (Get-Content (Join-Path $Bundles 'tuning_a_screen.zip.sha256')).Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not destroy the machine' } else { "HASH OK $got" }
```

**3A.5 On Vast.ai: DESTROY the old screening instance 58230725.** Only do this after the verified download and backup above. If it is still running, destroy it within one hour of closing billing; if already stopped, destroy it directly. This instance is not needed for confirmation.

### Machine B

**3B.1 On machine B: check and pack.** You should see: **`38/38 complete`**, then a file count with **no checkpoints**, then a long hash.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --stage screen --role b && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_b_screen.zip && sha256sum ~/tuning_b_screen.zip | tee ~/tuning_b_screen.zip.sha256
```
If it says less than 38/38, do not continue. Go to Part 9.

**3B.2 On laptop PowerShell: download.** You should see: **`HASH OK`**. If scp asks to trust the host, type `yes`.
```powershell
$Bundles = Join-Path $HOME 'Downloads'
scp -P 55203 root@108.240.82.27:tuning_b_screen.zip root@108.240.82.27:tuning_b_screen.zip.sha256 $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Download failed: do not stop the machine' }
$got = (Get-FileHash (Join-Path $Bundles 'tuning_b_screen.zip')).Hash.ToLower()
$want = (Get-Content (Join-Path $Bundles 'tuning_b_screen.zip.sha256')).Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not stop the machine' } else { "HASH OK $got" }
```

**3B.3 On machine B: close billing.** You should see: **`Stop the instance in the console within one hour`**.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role b --close
```

**3B.4 On laptop: back up and recheck.** Copy `tuning_b_screen.zip` and `tuning_b_screen.zip.sha256` from Downloads to a USB drive or cloud storage. Confirm the backup has finished. Recheck the Downloads copy; you should see **`HASH OK`**:
```powershell
$Bundles = Join-Path $HOME 'Downloads'
$got = (Get-FileHash (Join-Path $Bundles 'tuning_b_screen.zip')).Hash.ToLower()
$want = (Get-Content (Join-Path $Bundles 'tuning_b_screen.zip.sha256')).Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not destroy the machine' } else { "HASH OK $got" }
```

**3B.5 On Vast.ai: DESTROY the old screening instance 58230761.** Only do this after the verified download and backup above. If it is still running, destroy it within one hour of closing billing; if already stopped, destroy it directly. This instance is not needed for confirmation.

### Machine C

**3C.1 On machine C: check and pack.** You should see: **`29/29 complete`**, then a file count with **no checkpoints**, then a long hash.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --stage screen --role c && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_c_screen.zip && sha256sum ~/tuning_c_screen.zip | tee ~/tuning_c_screen.zip.sha256
```
If it says less than 29/29, do not continue. Go to Part 9.

**3C.2 On laptop PowerShell: download.** You should see: **`HASH OK`**. If scp asks to trust the host, type `yes`.
```powershell
$Bundles = Join-Path $HOME 'Downloads'
scp -P 40318 root@159.48.242.36:tuning_c_screen.zip root@159.48.242.36:tuning_c_screen.zip.sha256 $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Download failed: do not stop the machine' }
$got = (Get-FileHash (Join-Path $Bundles 'tuning_c_screen.zip')).Hash.ToLower()
$want = (Get-Content (Join-Path $Bundles 'tuning_c_screen.zip.sha256')).Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not stop the machine' } else { "HASH OK $got" }
```

**3C.3 On machine C: close billing.** You should see: **`Stop the instance in the console within one hour`**.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/billing_session.py --role c --close
```

**3C.4 On laptop: back up and recheck.** Copy `tuning_c_screen.zip` and `tuning_c_screen.zip.sha256` from Downloads to a USB drive or cloud storage. Confirm the backup has finished. Recheck the Downloads copy; you should see **`HASH OK`**:
```powershell
$Bundles = Join-Path $HOME 'Downloads'
$got = (Get-FileHash (Join-Path $Bundles 'tuning_c_screen.zip')).Hash.ToLower()
$want = (Get-Content (Join-Path $Bundles 'tuning_c_screen.zip.sha256')).Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not destroy the machine' } else { "HASH OK $got" }
```

**3C.5 On Vast.ai: DESTROY the old screening instance 58230733.** Only do this after the verified download and backup above. If it is still running, destroy it within one hour of closing billing; if already stopped, destroy it directly. This instance is not needed for confirmation.

If the machine remained running more than one hour after closing billing, note the extra time for the screening cost record. Confirmation does not carry forward a spending budget.

## Part 4. On the laptop, once all three screening bundles are saved

You need all three screening zips in Downloads: `tuning_a_screen.zip`, `tuning_b_screen.zip`, `tuning_c_screen.zip`, each with its `.sha256` file. Keep this PowerShell window open until Part 7, or paste 4.1 again in a new window.

**4.1 Set up variables.**
```powershell
$ErrorActionPreference = 'Stop'
$Repo = 'C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym'
$Python = 'D:\miniconda3\envs\temporal_gym\python.exe'
$ResultsRoot = Join-Path $Repo 'run_scripts\tuning\results'
$Mode = 'full'
$Bundles = Join-Path $HOME 'Downloads'
Set-Location -LiteralPath $Repo
```

**4.2 Merge the three zips.** This checks every hash again and refuses to overwrite any file that differs. You should see: **`BUNDLES VERIFIED AND MERGED`**.
<!-- rehearsal:screen-merge -->
```powershell
& "$Repo\run_scripts\tuning\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @((Join-Path $Bundles 'tuning_a_screen.zip'), (Join-Path $Bundles 'tuning_b_screen.zip'), (Join-Path $Bundles 'tuning_c_screen.zip'))
```

**4.3 Check screening and freeze the finalists** (the top two candidates per cell). You should see: **`88/88 complete`**, then **`Finalists frozen`**, then a 64-character hash. **Write this hash down**: it is the finalist hash used in Part 5.
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

**4.4 Commit and push only the finalists file.** You should see: the commit and push succeed, with no error.
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

## Part 5. Rent three new machines and start confirmation

Start Part 5 only after Part 4 has finished (finalists pushed). Confirmation uses **three new instances**, labelled **confirmation A**, **confirmation B**, and **confirmation C**. The role letters stay the same, but all instance IDs and SSH addresses are new. Do not run a command containing an unfilled `<...>` placeholder.

Choose the hardware and free-disk requirements in Part 10. Each new machine needs its own installation, W&B login and preflight. Keep the same training settings and one CPU thread. No screening rerun or upload of screening checkpoints is needed; each fresh clone gets the frozen `finalists.json` from GitHub. The screening evidence stays on the laptop for Part 7.

**No confirmation spending budget:** do not transfer screening billing files or carried hours. The launcher still requires a start timestamp. The first command on each new machine saves its current UTC time automatically; launch and restart commands reuse that saved value. You do not need to record, remember or send any times. Its existing `BUDGET WARNING` messages are advisory and never shorten or stop runs; leave confirmation running through `BATCH COMPLETE`. No overall spending cap is claimed for confirmation. Training budgets (environment steps), evaluations, seeds and the estimated run times remain unchanged.

| Confirmation role | New instance ID | SSH port | SSH host |
|---|---|---|---|
| A | `58344703` | `40284` | `35.131.20.10` |
| B | `58344678` | `33554` | `185.74.219.78` |
| C | `58344651` | `40354` | `159.48.242.36` |

### 5.0 Send me the three new SSH connection commands before setup

1. First complete **only 5A.1, 5B.1 and 5C.1** below: rent the three new machines. Do not proceed to their connection/setup steps yet.
2. In this chat, send me **all three SSH connection commands from Vast.ai** and their new instance IDs. No timestamps are needed. Use this format:

```text
Confirmation A
Instance ID: ...
SSH command: ssh -p ... root@...

Confirmation B
Instance ID: ...
SSH command: ssh -p ... root@...

Confirmation C
Instance ID: ...
SSH command: ssh -p ... root@...
```

The SSH connection commands provide the host and port; your private SSH key stays on your laptop.

3. I will fill in the placeholders in this guide. Once that update is ready, continue at **5A.2, 5B.2 and 5C.2** on the machines you already rented. **Do not rent them again.**

### New confirmation machine A (~2.5–3 h)

**5A.1 On Vast.ai: rent a new machine for role A.** Send its new instance ID and SSH connection command in step **5.0**. There is no time to record. Once the placeholders are filled in, continue at 5A.2.

**5A.2 On laptop PowerShell: connect to the NEW machine A.** Fill in its SSH port and host from Vast.ai. You should see its new instance prompt:
```powershell
ssh -p '40284' 'root@35.131.20.10'
```

**5A.3 FIRST command on new machine A: save the current UTC time automatically, then start tmux.** Run this block immediately after connecting. It saves the time in `~/tuning_a_confirm_start.txt` and preserves it if the block is repeated. You should see **`Confirmation start saved:`** and the time, then a tmux status bar. You do not need to write down the time:
```bash
if [ ! -s ~/tuning_a_confirm_start.txt ]; then date -u +%Y-%m-%dT%H:%M:%SZ > ~/tuning_a_confirm_start.txt; fi
printf 'Confirmation start saved: '
cat ~/tuning_a_confirm_start.txt
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_a; fi
```

**5A.4 On new machine A: download the code and finalists.** You should see the clone finish without errors:
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```

**5A.5 On new machine A: check code and finalist hash.** You should see **`CODE CHECK DONE`**, then a hash **exactly matching step 4.3**. If either check fails, stop before training:
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --exit-code fe837314e40662649e0fe1bbfb7d88cfe2307d5b HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && git diff --exit-code HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && echo "CODE CHECK DONE" && sha256sum run_scripts/tuning/results/v3/full/finalists.json
```

**5A.6 On new machine A: install Python packages.** You should see a Python version, then pip finishing without errors:
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in to W&B yourself. You should see a login success message:
```bash
~/tcg/bin/wandb login
```

**5A.7 On new machine A: run preflight.** It must end with **`PREFLIGHT PASSED`**. Fix every FAIL before starting:
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/preflight.py --role a --device-policy plan --wandb-mode online --learning-check 2>&1 | tee ~/preflight_a.log
```

**5A.8 On new machine A: start confirmation (6 runs).** This automatically reads the time saved by your first command in 5A.3; no timestamp needs filling in. You should see `START ...` and `COMPLETE ...` lines, ending with **`BATCH COMPLETE`**:
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_a_tuning.sh --python ~/tcg/bin/python --stage confirm --wandb-mode online --billing-start "$(cat ~/tuning_a_confirm_start.txt)" 2>&1 | tee -a ~/tuning_a.log
```

### New confirmation machine B (~12–15 h)

**5B.1 On Vast.ai: rent a new machine for role B.** Send its new instance ID and SSH connection command in step **5.0**. There is no time to record. Once the placeholders are filled in, continue at 5B.2.

**5B.2 On laptop PowerShell: connect to the NEW machine B.** Fill in its SSH port and host from Vast.ai. You should see its new instance prompt:
```powershell
ssh -p '33554' 'root@185.74.219.78'
```

**5B.3 FIRST command on new machine B: save the current UTC time automatically, then start tmux.** Run this block immediately after connecting. It saves the time in `~/tuning_b_confirm_start.txt` and preserves it if the block is repeated. You should see **`Confirmation start saved:`** and the time, then a tmux status bar. You do not need to write down the time:
```bash
if [ ! -s ~/tuning_b_confirm_start.txt ]; then date -u +%Y-%m-%dT%H:%M:%SZ > ~/tuning_b_confirm_start.txt; fi
printf 'Confirmation start saved: '
cat ~/tuning_b_confirm_start.txt
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_b; fi
```

**5B.4 On new machine B: download the code and finalists.** You should see the clone finish without errors:
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```

**5B.5 On new machine B: check code and finalist hash.** You should see **`CODE CHECK DONE`**, then a hash **exactly matching step 4.3**. If either check fails, stop before training:
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --exit-code fe837314e40662649e0fe1bbfb7d88cfe2307d5b HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && git diff --exit-code HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && echo "CODE CHECK DONE" && sha256sum run_scripts/tuning/results/v3/full/finalists.json
```

**5B.6 On new machine B: install Python packages.** You should see a Python version, then pip finishing without errors:
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in to W&B yourself. You should see a login success message:
```bash
~/tcg/bin/wandb login
```

**5B.7 On new machine B: run preflight.** It must end with **`PREFLIGHT PASSED`**. Fix every FAIL before starting:
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/preflight.py --role b --device-policy plan --wandb-mode online --learning-check 2>&1 | tee ~/preflight_b.log
```

**5B.8 On new machine B: start confirmation (36 runs).** This automatically reads the time saved by your first command in 5B.3; no timestamp needs filling in. You should see `START ...` and `COMPLETE ...` lines, ending with **`BATCH COMPLETE`**:
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_b_tuning.sh --python ~/tcg/bin/python --stage confirm --wandb-mode online --billing-start "$(cat ~/tuning_b_confirm_start.txt)" 2>&1 | tee -a ~/tuning_b.log
```

### New confirmation machine C (~8–9 h)

**5C.1 On Vast.ai: rent a new machine for role C.** Send its new instance ID and SSH connection command in step **5.0**. There is no time to record. Once the placeholders are filled in, continue at 5C.2.

**5C.2 On laptop PowerShell: connect to the NEW machine C.** Fill in its SSH port and host from Vast.ai. You should see its new instance prompt:
```powershell
ssh -p '40354' 'root@159.48.242.36'
```

**5C.3 FIRST command on new machine C: save the current UTC time automatically, then start tmux.** Run this block immediately after connecting. It saves the time in `~/tuning_c_confirm_start.txt` and preserves it if the block is repeated. You should see **`Confirmation start saved:`** and the time, then a tmux status bar. You do not need to write down the time:
```bash
if [ ! -s ~/tuning_c_confirm_start.txt ]; then date -u +%Y-%m-%dT%H:%M:%SZ > ~/tuning_c_confirm_start.txt; fi
printf 'Confirmation start saved: '
cat ~/tuning_c_confirm_start.txt
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s tuning_c; fi
```

**5C.4 On new machine C: download the code and finalists.** You should see the clone finish without errors:
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym
```

**5C.5 On new machine C: check code and finalist hash.** You should see **`CODE CHECK DONE`**, then a hash **exactly matching step 4.3**. If either check fails, stop before training:
```bash
cd ~/temporal-composition-gym && git log -1 --oneline && git diff --exit-code fe837314e40662649e0fe1bbfb7d88cfe2307d5b HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && git diff --exit-code HEAD -- '*.py' '*.sh' '*.ps1' requirements.txt setup.py run_scripts/tuning/trials_v3.json run_scripts/baseline/baseline_plan.json && echo "CODE CHECK DONE" && sha256sum run_scripts/tuning/results/v3/full/finalists.json
```

**5C.6 On new machine C: install Python packages.** You should see a Python version, then pip finishing without errors:
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```
Log in to W&B yourself. You should see a login success message:
```bash
~/tcg/bin/wandb login
```

**5C.7 On new machine C: run preflight.** It must end with **`PREFLIGHT PASSED`**. Fix every FAIL before starting:
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/preflight.py --role c --device-policy plan --wandb-mode online --learning-check 2>&1 | tee ~/preflight_c.log
```

**5C.8 On new machine C: start confirmation (24 runs).** This automatically reads the time saved by your first command in 5C.3; no timestamp needs filling in. You should see `START ...` and `COMPLETE ...` lines, ending with **`BATCH COMPLETE`**:
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_c_tuning.sh --python ~/tcg/bin/python --stage confirm --wandb-mode online --billing-start "$(cat ~/tuning_c_confirm_start.txt)" 2>&1 | tee -a ~/tuning_c.log
```

Check on each confirmation machine the same way as in Part 2. If interrupted, use its confirmation restart command in Part 9. It reads the original saved time automatically, including after reconnecting or opening a new tmux window. Keep `~/tuning_a_confirm_start.txt` (or b/c) unchanged until that machine is finished.

## Part 6. Finish and DESTROY each confirmation machine independently

**Start this part for a machine as soon as it prints `BATCH COMPLETE`. Do not wait for the other machines.** Follow all six steps under that machine, ending with **DESTROY**. Part 7 is a later, laptop-only step and is not required before destroying a machine that has passed its own checks here.

Each new machine contains its confirmation runs. Its expected count is **A 6/6, B 36/36, C 24/24**. Screening results stay on the laptop; Part 7 checks the combined totals. There is no confirmation spending budget, billing-close or carry-over step.

**Model downloads are required at 6A.5 / 6B.5 / 6C.5 before destruction.** Model archives are created in the rented machine's home directory and downloaded to **`$HOME\Downloads\tuning-confirmation-models`** on your laptop, outside this repository. Keep them there and make your own external backups. Do not extract models into this checkout, copy them into the results folder, pass them to `merge_bundles.ps1`, or stage/commit/upload them to GitHub. The CSV/JSON results use the existing separate collection steps.

Estimated final-model download sizes: **A ~15 MB, B ~70 MB, C ~400 MB; ~485 MB total** before additional archive compression. Restart/replay data is excluded.

| Finished machine | Follow now | Destroy at | Can the others still run? |
|---|---|---|---|
| Confirmation A | 6A.1–6A.6 | **6A.6** | **Yes** |
| Confirmation B | 6B.1–6B.6 | **6B.6** | **Yes** |
| Confirmation C | 6C.1–6C.6 | **6C.6** | **Yes** |

### New confirmation machine A: collect, verify, back up, DESTROY

**6A.1 On machine A: check and pack.** You should see **`6/6 complete`**, then a file count with **no checkpoints**, then a long hash. If the check fails, keep this machine and investigate:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --stage confirm --role a && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_a_confirm.zip && sha256sum ~/tuning_a_confirm.zip | tee ~/tuning_a_confirm.zip.sha256
```

**6A.2 On laptop PowerShell: download and verify.** Use this new machine's SSH details. You should see **`HASH OK`**. If anything fails, do not destroy it:
```powershell
$ErrorActionPreference = 'Stop'
$Bundles = Join-Path $HOME 'Downloads'
scp -P '40284' 'root@35.131.20.10:tuning_a_confirm.zip' 'root@35.131.20.10:tuning_a_confirm.zip.sha256' $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Download failed: do not destroy the machine' }
$got = (Get-FileHash (Join-Path $Bundles 'tuning_a_confirm.zip')).Hash.ToLower()
$want = (Get-Content (Join-Path $Bundles 'tuning_a_confirm.zip.sha256')).Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not destroy the machine' } else { "HASH OK $got" }
```

**6A.3 On laptop PowerShell: merge and verify A's confirmation results.** Use the variables from **4.1**; paste 4.1 again first if this is a new PowerShell window. This step needs only A's zip. You should see **`BUNDLES VERIFIED AND MERGED`**, then **`6/6 complete`**, then **`CONFIRMATION A VERIFIED ON LAPTOP`**. Other machines may still show missing or unfinished entries; only A's confirmation count must be complete:
<!-- rehearsal:confirm-a-collect -->
```powershell
& "$Repo\run_scripts\tuning\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @((Join-Path $Bundles 'tuning_a_confirm.zip'))
& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --results-root $ResultsRoot --mode $Mode --stage confirm --role a
if ($LASTEXITCODE -ne 0) { throw 'Confirmation A incomplete on laptop: do not destroy the machine' }
'CONFIRMATION A VERIFIED ON LAPTOP'
```

**6A.4 On laptop: back up A's zip and checksum off the laptop.** Copy `tuning_a_confirm.zip` and `tuning_a_confirm.zip.sha256` from Downloads to a USB drive or cloud storage. Wait for the copy/upload to finish. Replace `<A_CONFIRM_BACKUP_FOLDER>` below with the local path to that backup folder (a USB path or a locally readable cloud-synced folder). You should see **`BACKUP VERIFIED`**. If verification fails, keep the machine:
```powershell
$ErrorActionPreference = 'Stop'
$Bundles = Join-Path $HOME 'Downloads'
$BackupFolder = '<A_CONFIRM_BACKUP_FOLDER>'
$want = (Get-Content (Join-Path $Bundles 'tuning_a_confirm.zip.sha256')).Split(' ')[0]
$backupHash = (Get-FileHash (Join-Path $BackupFolder 'tuning_a_confirm.zip')).Hash.ToLower()
$backupWant = (Get-Content (Join-Path $BackupFolder 'tuning_a_confirm.zip.sha256')).Split(' ')[0]
if ($backupHash -ne $want -or $backupWant -ne $want) { throw 'BACKUP MISMATCH: do not destroy the machine' } else { 'BACKUP VERIFIED' }
```

**6A.5 REQUIRED: download A's final models and make your own backup before destroying it.** This is separate from the CSV/JSON results. Follow **6A.5a–6A.5c** in order.

**6A.5a On machine A: pack the final models outside the checkout.** You should see **`6 final policies bundled`** and a hash. This selects the verified completed confirmation attempts, includes one final policy and its config/identity/metadata/summary per run, and excludes restart/replay files. The archive is saved at `~/tuning_a_confirm_models.tar.gz`. It refuses to overwrite an existing archive; if you already packed it successfully, reuse it and continue to 6A.5b.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B - <<'PY'
from pathlib import Path
import hashlib
import json
import tarfile
from agents.common.tuning_plan import load_plan
from agents.common.tuning_records import manifest

root = Path('run_scripts/tuning/results').resolve()
role = 'a'
report = manifest(root, load_plan(), 'full')
runs = [r for r in report['entries'] if r['stage'] == 'confirm' and r['role'] == role]
if len(runs) != 6 or any(r['state'] != 'complete' for r in runs) or report['unexpected_runs']:
    raise SystemExit('Confirmation verification failed: keep the machine')
files = []
for run in runs:
    folder = root / run['run_dir']
    step = json.loads((folder / 'final_summary.json').read_text())['global_step']
    models = list((folder / 'checkpoints').glob(f'policy_step{step:09d}.*'))
    if len(models) != 1 or models[0].suffix not in ('.pt', '.zip'):
        raise SystemExit(f'Missing or ambiguous final policy: {folder}; keep the machine')
    files.append(models[0])
    files.extend(folder / name for name in ('identity.json', 'config.resolved.json', 'metadata.json', 'final_summary.json'))
out = Path.home() / f'tuning_{role}_confirm_models.tar.gz'
with tarfile.open(out, 'x:gz') as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
hash_value = hashlib.sha256()
with out.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        hash_value.update(chunk)
Path(str(out) + '.sha256').write_text(f'{hash_value.hexdigest()}  {out.name}\n', encoding='utf-8')
print(f'{len(runs)} final policies bundled in {out} ({out.stat().st_size / 1e6:.1f} MB); no replay/restart data')
print(hash_value.hexdigest())
PY
```

**6A.5b On laptop PowerShell: download models OUTSIDE the repository and verify.** You should see **`MODEL HASH OK`**. This creates `$HOME\Downloads\tuning-confirmation-models`; model files stay in this separate folder. If the download files already exist, keep them and verify the existing download instead of overwriting it.
```powershell
$ErrorActionPreference = 'Stop'
$ModelDownloads = [IO.Path]::GetFullPath((Join-Path $HOME 'Downloads\tuning-confirmation-models'))
$Checkout = [IO.Path]::GetFullPath('C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym')
if ($ModelDownloads -eq $Checkout -or $ModelDownloads.StartsWith($Checkout + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Models must be downloaded outside the repository' }
New-Item -ItemType Directory -Path $ModelDownloads -Force | Out-Null
$ModelArchive = Join-Path $ModelDownloads 'tuning_a_confirm_models.tar.gz'
if ((Test-Path -LiteralPath $ModelArchive) -or (Test-Path -LiteralPath "$ModelArchive.sha256")) { throw 'Model download already exists: keep it and run only the hash-verification lines below' }
scp -P '40284' 'root@35.131.20.10:tuning_a_confirm_models.tar.gz' 'root@35.131.20.10:tuning_a_confirm_models.tar.gz.sha256' $ModelDownloads
if ($LASTEXITCODE -ne 0) { throw 'Model download failed: do not destroy the machine' }
$got = (Get-FileHash -LiteralPath $ModelArchive).Hash.ToLower()
$want = (Get-Content -LiteralPath "$ModelArchive.sha256").Split(' ')[0]
if ($got -ne $want) { throw 'MODEL HASH MISMATCH: do not destroy the machine' } else { "MODEL HASH OK $got" }
```

**6A.5c Back up the models yourself, then verify the backup.** Copy `tuning_a_confirm_models.tar.gz` and its `.sha256` file from `$HOME\Downloads\tuning-confirmation-models` to the external backup folder you chose in **6A.4**. Wait for your copy/upload to finish. In the same PowerShell window, run the following; `$BackupFolder` is the folder from 6A.4. You should see **`MODEL BACKUP VERIFIED`**. Keep both the Downloads copy and your external backup outside the repository. You can keep the archive packed; any later extraction must also go outside the repository.
```powershell
$want = (Get-Content -LiteralPath "$ModelArchive.sha256").Split(' ')[0]
$backupModelHash = (Get-FileHash (Join-Path $BackupFolder 'tuning_a_confirm_models.tar.gz')).Hash.ToLower()
$backupModelWant = (Get-Content (Join-Path $BackupFolder 'tuning_a_confirm_models.tar.gz.sha256')).Split(' ')[0]
if ($backupModelHash -ne $want -or $backupModelWant -ne $want) { throw 'MODEL BACKUP MISMATCH: do not destroy the machine' } else { 'MODEL BACKUP VERIFIED' }
```

If you also want raw stdout/stderr or W&B diagnostic logs, copy them while machine A is still running. These logs are optional and are not in the model archive.

**6A.6 NOW on Vast.ai: DESTROY confirmation A, instance `58344703`.** Do this only after the machine and laptop checks pass, **`BACKUP VERIFIED`**, **`MODEL HASH OK`** and **`MODEL BACKUP VERIFIED`** appear, and any optional log copies are finished. Choose **Destroy**. **You can do this immediately even if the other two machines are still running; do not wait for Part 7 or Part 8.** This is the new confirmation instance recorded in Part 5.

### New confirmation machine B: collect, verify, back up, DESTROY

**6B.1 On machine B: check and pack.** You should see **`36/36 complete`**, then a file count with **no checkpoints**, then a long hash. If the check fails, keep this machine and investigate:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --stage confirm --role b && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_b_confirm.zip && sha256sum ~/tuning_b_confirm.zip | tee ~/tuning_b_confirm.zip.sha256
```

**6B.2 On laptop PowerShell: download and verify.** Use this new machine's SSH details. You should see **`HASH OK`**. If anything fails, do not destroy it:
```powershell
$ErrorActionPreference = 'Stop'
$Bundles = Join-Path $HOME 'Downloads'
scp -P '33554' 'root@185.74.219.78:tuning_b_confirm.zip' 'root@185.74.219.78:tuning_b_confirm.zip.sha256' $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Download failed: do not destroy the machine' }
$got = (Get-FileHash (Join-Path $Bundles 'tuning_b_confirm.zip')).Hash.ToLower()
$want = (Get-Content (Join-Path $Bundles 'tuning_b_confirm.zip.sha256')).Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not destroy the machine' } else { "HASH OK $got" }
```

**6B.3 On laptop PowerShell: merge and verify B's confirmation results.** Use the variables from **4.1**; paste 4.1 again first if this is a new PowerShell window. This step needs only B's zip. You should see **`BUNDLES VERIFIED AND MERGED`**, then **`36/36 complete`**, then **`CONFIRMATION B VERIFIED ON LAPTOP`**. Other machines may still show missing or unfinished entries; only B's confirmation count must be complete:
<!-- rehearsal:confirm-b-collect -->
```powershell
& "$Repo\run_scripts\tuning\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @((Join-Path $Bundles 'tuning_b_confirm.zip'))
& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --results-root $ResultsRoot --mode $Mode --stage confirm --role b
if ($LASTEXITCODE -ne 0) { throw 'Confirmation B incomplete on laptop: do not destroy the machine' }
'CONFIRMATION B VERIFIED ON LAPTOP'
```

**6B.4 On laptop: back up B's zip and checksum off the laptop.** Copy `tuning_b_confirm.zip` and `tuning_b_confirm.zip.sha256` from Downloads to a USB drive or cloud storage. Wait for the copy/upload to finish. Replace `<B_CONFIRM_BACKUP_FOLDER>` below with the local path to that backup folder (a USB path or a locally readable cloud-synced folder). You should see **`BACKUP VERIFIED`**. If verification fails, keep the machine:
```powershell
$ErrorActionPreference = 'Stop'
$Bundles = Join-Path $HOME 'Downloads'
$BackupFolder = '<B_CONFIRM_BACKUP_FOLDER>'
$want = (Get-Content (Join-Path $Bundles 'tuning_b_confirm.zip.sha256')).Split(' ')[0]
$backupHash = (Get-FileHash (Join-Path $BackupFolder 'tuning_b_confirm.zip')).Hash.ToLower()
$backupWant = (Get-Content (Join-Path $BackupFolder 'tuning_b_confirm.zip.sha256')).Split(' ')[0]
if ($backupHash -ne $want -or $backupWant -ne $want) { throw 'BACKUP MISMATCH: do not destroy the machine' } else { 'BACKUP VERIFIED' }
```

**6B.5 REQUIRED: download B's final models and make your own backup before destroying it.** This is separate from the CSV/JSON results. Follow **6B.5a–6B.5c** in order.

**6B.5a On machine B: pack the final models outside the checkout.** You should see **`36 final policies bundled`** and a hash. This selects the verified completed confirmation attempts, includes one final policy and its config/identity/metadata/summary per run, and excludes restart/replay files. The archive is saved at `~/tuning_b_confirm_models.tar.gz`. It refuses to overwrite an existing archive; if you already packed it successfully, reuse it and continue to 6B.5b.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B - <<'PY'
from pathlib import Path
import hashlib
import json
import tarfile
from agents.common.tuning_plan import load_plan
from agents.common.tuning_records import manifest

root = Path('run_scripts/tuning/results').resolve()
role = 'b'
report = manifest(root, load_plan(), 'full')
runs = [r for r in report['entries'] if r['stage'] == 'confirm' and r['role'] == role]
if len(runs) != 36 or any(r['state'] != 'complete' for r in runs) or report['unexpected_runs']:
    raise SystemExit('Confirmation verification failed: keep the machine')
files = []
for run in runs:
    folder = root / run['run_dir']
    step = json.loads((folder / 'final_summary.json').read_text())['global_step']
    models = list((folder / 'checkpoints').glob(f'policy_step{step:09d}.*'))
    if len(models) != 1 or models[0].suffix not in ('.pt', '.zip'):
        raise SystemExit(f'Missing or ambiguous final policy: {folder}; keep the machine')
    files.append(models[0])
    files.extend(folder / name for name in ('identity.json', 'config.resolved.json', 'metadata.json', 'final_summary.json'))
out = Path.home() / f'tuning_{role}_confirm_models.tar.gz'
with tarfile.open(out, 'x:gz') as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
hash_value = hashlib.sha256()
with out.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        hash_value.update(chunk)
Path(str(out) + '.sha256').write_text(f'{hash_value.hexdigest()}  {out.name}\n', encoding='utf-8')
print(f'{len(runs)} final policies bundled in {out} ({out.stat().st_size / 1e6:.1f} MB); no replay/restart data')
print(hash_value.hexdigest())
PY
```

**6B.5b On laptop PowerShell: download models OUTSIDE the repository and verify.** You should see **`MODEL HASH OK`**. This creates `$HOME\Downloads\tuning-confirmation-models`; model files stay in this separate folder. If the download files already exist, keep them and verify the existing download instead of overwriting it.
```powershell
$ErrorActionPreference = 'Stop'
$ModelDownloads = [IO.Path]::GetFullPath((Join-Path $HOME 'Downloads\tuning-confirmation-models'))
$Checkout = [IO.Path]::GetFullPath('C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym')
if ($ModelDownloads -eq $Checkout -or $ModelDownloads.StartsWith($Checkout + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Models must be downloaded outside the repository' }
New-Item -ItemType Directory -Path $ModelDownloads -Force | Out-Null
$ModelArchive = Join-Path $ModelDownloads 'tuning_b_confirm_models.tar.gz'
if ((Test-Path -LiteralPath $ModelArchive) -or (Test-Path -LiteralPath "$ModelArchive.sha256")) { throw 'Model download already exists: keep it and run only the hash-verification lines below' }
scp -P '33554' 'root@185.74.219.78:tuning_b_confirm_models.tar.gz' 'root@185.74.219.78:tuning_b_confirm_models.tar.gz.sha256' $ModelDownloads
if ($LASTEXITCODE -ne 0) { throw 'Model download failed: do not destroy the machine' }
$got = (Get-FileHash -LiteralPath $ModelArchive).Hash.ToLower()
$want = (Get-Content -LiteralPath "$ModelArchive.sha256").Split(' ')[0]
if ($got -ne $want) { throw 'MODEL HASH MISMATCH: do not destroy the machine' } else { "MODEL HASH OK $got" }
```

**6B.5c Back up the models yourself, then verify the backup.** Copy `tuning_b_confirm_models.tar.gz` and its `.sha256` file from `$HOME\Downloads\tuning-confirmation-models` to the external backup folder you chose in **6B.4**. Wait for your copy/upload to finish. In the same PowerShell window, run the following; `$BackupFolder` is the folder from 6B.4. You should see **`MODEL BACKUP VERIFIED`**. Keep both the Downloads copy and your external backup outside the repository. You can keep the archive packed; any later extraction must also go outside the repository.
```powershell
$want = (Get-Content -LiteralPath "$ModelArchive.sha256").Split(' ')[0]
$backupModelHash = (Get-FileHash (Join-Path $BackupFolder 'tuning_b_confirm_models.tar.gz')).Hash.ToLower()
$backupModelWant = (Get-Content (Join-Path $BackupFolder 'tuning_b_confirm_models.tar.gz.sha256')).Split(' ')[0]
if ($backupModelHash -ne $want -or $backupModelWant -ne $want) { throw 'MODEL BACKUP MISMATCH: do not destroy the machine' } else { 'MODEL BACKUP VERIFIED' }
```

If you also want raw stdout/stderr or W&B diagnostic logs, copy them while machine B is still running. These logs are optional and are not in the model archive.

**6B.6 NOW on Vast.ai: DESTROY confirmation B, instance `58344678`.** Do this only after the machine and laptop checks pass, **`BACKUP VERIFIED`**, **`MODEL HASH OK`** and **`MODEL BACKUP VERIFIED`** appear, and any optional log copies are finished. Choose **Destroy**. **You can do this immediately even if the other two machines are still running; do not wait for Part 7 or Part 8.** This is the new confirmation instance recorded in Part 5.

### New confirmation machine C: collect, verify, back up, DESTROY

**6C.1 On machine C: check and pack.** You should see **`24/24 complete`**, then a file count with **no checkpoints**, then a long hash. If the check fails, keep this machine and investigate:
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/tuning/check_manifest.py --stage confirm --role c && ~/tcg/bin/python -B run_scripts/tuning/collect.py --out ~/tuning_c_confirm.zip && sha256sum ~/tuning_c_confirm.zip | tee ~/tuning_c_confirm.zip.sha256
```

**6C.2 On laptop PowerShell: download and verify.** Use this new machine's SSH details. You should see **`HASH OK`**. If anything fails, do not destroy it:
```powershell
$ErrorActionPreference = 'Stop'
$Bundles = Join-Path $HOME 'Downloads'
scp -P '40354' 'root@159.48.242.36:tuning_c_confirm.zip' 'root@159.48.242.36:tuning_c_confirm.zip.sha256' $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Download failed: do not destroy the machine' }
$got = (Get-FileHash (Join-Path $Bundles 'tuning_c_confirm.zip')).Hash.ToLower()
$want = (Get-Content (Join-Path $Bundles 'tuning_c_confirm.zip.sha256')).Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not destroy the machine' } else { "HASH OK $got" }
```

**6C.3 On laptop PowerShell: merge and verify C's confirmation results.** Use the variables from **4.1**; paste 4.1 again first if this is a new PowerShell window. This step needs only C's zip. You should see **`BUNDLES VERIFIED AND MERGED`**, then **`24/24 complete`**, then **`CONFIRMATION C VERIFIED ON LAPTOP`**. Other machines may still show missing or unfinished entries; only C's confirmation count must be complete:
<!-- rehearsal:confirm-c-collect -->
```powershell
& "$Repo\run_scripts\tuning\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @((Join-Path $Bundles 'tuning_c_confirm.zip'))
& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --results-root $ResultsRoot --mode $Mode --stage confirm --role c
if ($LASTEXITCODE -ne 0) { throw 'Confirmation C incomplete on laptop: do not destroy the machine' }
'CONFIRMATION C VERIFIED ON LAPTOP'
```

**6C.4 On laptop: back up C's zip and checksum off the laptop.** Copy `tuning_c_confirm.zip` and `tuning_c_confirm.zip.sha256` from Downloads to a USB drive or cloud storage. Wait for the copy/upload to finish. Replace `<C_CONFIRM_BACKUP_FOLDER>` below with the local path to that backup folder (a USB path or a locally readable cloud-synced folder). You should see **`BACKUP VERIFIED`**. If verification fails, keep the machine:
```powershell
$ErrorActionPreference = 'Stop'
$Bundles = Join-Path $HOME 'Downloads'
$BackupFolder = '<C_CONFIRM_BACKUP_FOLDER>'
$want = (Get-Content (Join-Path $Bundles 'tuning_c_confirm.zip.sha256')).Split(' ')[0]
$backupHash = (Get-FileHash (Join-Path $BackupFolder 'tuning_c_confirm.zip')).Hash.ToLower()
$backupWant = (Get-Content (Join-Path $BackupFolder 'tuning_c_confirm.zip.sha256')).Split(' ')[0]
if ($backupHash -ne $want -or $backupWant -ne $want) { throw 'BACKUP MISMATCH: do not destroy the machine' } else { 'BACKUP VERIFIED' }
```

**6C.5 REQUIRED: download C's final models and make your own backup before destroying it.** This is separate from the CSV/JSON results. Follow **6C.5a–6C.5c** in order.

**6C.5a On machine C: pack the final models outside the checkout.** You should see **`24 final policies bundled`** and a hash. This selects the verified completed confirmation attempts, includes one final policy and its config/identity/metadata/summary per run, and excludes restart/replay files. The archive is saved at `~/tuning_c_confirm_models.tar.gz`. It refuses to overwrite an existing archive; if you already packed it successfully, reuse it and continue to 6C.5b.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B - <<'PY'
from pathlib import Path
import hashlib
import json
import tarfile
from agents.common.tuning_plan import load_plan
from agents.common.tuning_records import manifest

root = Path('run_scripts/tuning/results').resolve()
role = 'c'
report = manifest(root, load_plan(), 'full')
runs = [r for r in report['entries'] if r['stage'] == 'confirm' and r['role'] == role]
if len(runs) != 24 or any(r['state'] != 'complete' for r in runs) or report['unexpected_runs']:
    raise SystemExit('Confirmation verification failed: keep the machine')
files = []
for run in runs:
    folder = root / run['run_dir']
    step = json.loads((folder / 'final_summary.json').read_text())['global_step']
    models = list((folder / 'checkpoints').glob(f'policy_step{step:09d}.*'))
    if len(models) != 1 or models[0].suffix not in ('.pt', '.zip'):
        raise SystemExit(f'Missing or ambiguous final policy: {folder}; keep the machine')
    files.append(models[0])
    files.extend(folder / name for name in ('identity.json', 'config.resolved.json', 'metadata.json', 'final_summary.json'))
out = Path.home() / f'tuning_{role}_confirm_models.tar.gz'
with tarfile.open(out, 'x:gz') as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
hash_value = hashlib.sha256()
with out.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        hash_value.update(chunk)
Path(str(out) + '.sha256').write_text(f'{hash_value.hexdigest()}  {out.name}\n', encoding='utf-8')
print(f'{len(runs)} final policies bundled in {out} ({out.stat().st_size / 1e6:.1f} MB); no replay/restart data')
print(hash_value.hexdigest())
PY
```

**6C.5b On laptop PowerShell: download models OUTSIDE the repository and verify.** You should see **`MODEL HASH OK`**. This creates `$HOME\Downloads\tuning-confirmation-models`; model files stay in this separate folder. If the download files already exist, keep them and verify the existing download instead of overwriting it.
```powershell
$ErrorActionPreference = 'Stop'
$ModelDownloads = [IO.Path]::GetFullPath((Join-Path $HOME 'Downloads\tuning-confirmation-models'))
$Checkout = [IO.Path]::GetFullPath('C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym')
if ($ModelDownloads -eq $Checkout -or $ModelDownloads.StartsWith($Checkout + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Models must be downloaded outside the repository' }
New-Item -ItemType Directory -Path $ModelDownloads -Force | Out-Null
$ModelArchive = Join-Path $ModelDownloads 'tuning_c_confirm_models.tar.gz'
if ((Test-Path -LiteralPath $ModelArchive) -or (Test-Path -LiteralPath "$ModelArchive.sha256")) { throw 'Model download already exists: keep it and run only the hash-verification lines below' }
scp -P '40354' 'root@159.48.242.36:tuning_c_confirm_models.tar.gz' 'root@159.48.242.36:tuning_c_confirm_models.tar.gz.sha256' $ModelDownloads
if ($LASTEXITCODE -ne 0) { throw 'Model download failed: do not destroy the machine' }
$got = (Get-FileHash -LiteralPath $ModelArchive).Hash.ToLower()
$want = (Get-Content -LiteralPath "$ModelArchive.sha256").Split(' ')[0]
if ($got -ne $want) { throw 'MODEL HASH MISMATCH: do not destroy the machine' } else { "MODEL HASH OK $got" }
```

**6C.5c Back up the models yourself, then verify the backup.** Copy `tuning_c_confirm_models.tar.gz` and its `.sha256` file from `$HOME\Downloads\tuning-confirmation-models` to the external backup folder you chose in **6C.4**. Wait for your copy/upload to finish. In the same PowerShell window, run the following; `$BackupFolder` is the folder from 6C.4. You should see **`MODEL BACKUP VERIFIED`**. Keep both the Downloads copy and your external backup outside the repository. You can keep the archive packed; any later extraction must also go outside the repository.
```powershell
$want = (Get-Content -LiteralPath "$ModelArchive.sha256").Split(' ')[0]
$backupModelHash = (Get-FileHash (Join-Path $BackupFolder 'tuning_c_confirm_models.tar.gz')).Hash.ToLower()
$backupModelWant = (Get-Content (Join-Path $BackupFolder 'tuning_c_confirm_models.tar.gz.sha256')).Split(' ')[0]
if ($backupModelHash -ne $want -or $backupModelWant -ne $want) { throw 'MODEL BACKUP MISMATCH: do not destroy the machine' } else { 'MODEL BACKUP VERIFIED' }
```

If you also want raw stdout/stderr or W&B diagnostic logs, copy them while machine C is still running. These logs are optional and are not in the model archive.

**6C.6 NOW on Vast.ai: DESTROY confirmation C, instance `58344651`.** Do this only after the machine and laptop checks pass, **`BACKUP VERIFIED`**, **`MODEL HASH OK`** and **`MODEL BACKUP VERIFIED`** appear, and any optional log copies are finished. Choose **Destroy**. **You can do this immediately even if the other two machines are still running; do not wait for Part 7 or Part 8.** This is the new confirmation instance recorded in Part 5.

After destroying a finished confirmation machine, leave the remaining machines running. Repeat their own Part 6 steps when they finish. Once all three confirmation result zips and all three model archives are downloaded, verified and backed up, continue to Part 7 on your laptop. Keep the model archives outside the repository throughout Part 7.

## Part 7. On the laptop, once all three confirmation zips are downloaded

Use the variables from step 4.1 (paste 4.1 again if this is a new PowerShell window). **Each confirmation machine can already be destroyed after its own Part 6 steps. Everything in Part 7 runs on the laptop and needs no rented machine.**

**7.1 Recheck and merge all three bundles.** This safely accepts the identical files already merged one machine at a time in Part 6. It does not require the machines to exist. You should see **`BUNDLES VERIFIED AND MERGED`**.
<!-- rehearsal:final-merge -->
```powershell
& "$Repo\run_scripts\tuning\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @((Join-Path $Bundles 'tuning_a_confirm.zip'), (Join-Path $Bundles 'tuning_b_confirm.zip'), (Join-Path $Bundles 'tuning_c_confirm.zip'))
```

**7.2 Check everything and select the final configs.** You should see: **`154/154 complete`**, a role list with A 27 / B 74 / C 53, then **`{"complete": true, "problems": []}`**.
<!-- rehearsal:final-check -->
```powershell
& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --results-root $ResultsRoot --mode $Mode
if ($LASTEXITCODE -ne 0) { throw 'Final manifest incomplete' }
$m = (& $Python -B "$Repo\run_scripts\tuning\check_manifest.py" --results-root $ResultsRoot --mode $Mode --json | ConvertFrom-Json)
$m.by_role | Format-List
& $Python -B "$Repo\run_scripts\tuning\select_configs.py" --results-root $ResultsRoot --mode $Mode
if ($LASTEXITCODE -ne 0) { throw 'Final selection failed' }
```

**7.3 Check the backups** of all six zips and their `.sha256` files somewhere off the laptop (for example cloud storage or a USB drive). The screening backups were made in Part 3; each confirmation backup was verified in Part 6 before destroying that machine. Also check that your three separate model archives and their checksums are backed up. Model archives stay outside the repository and are not merged, staged or published.

**7.4 Read** `run_scripts/tuning/results/v3/full/selection_report.md`. Note any deviations there, including that all runs used 1 CPU thread (the protocol planned 4).

**7.5 Stage the results.** This stages only CSV/JSON files and the report. The downloaded model archives and their backups remain outside the repository; do not add them to this step. You should see: a list of tuning CSV/JSON files plus the report, including `finalists.json` and `selected_configs.json`, and no checkpoints or `.pyc` files.
```powershell
Set-Location -LiteralPath $Repo
git restore --staged .
@("$ResultsRoot\v3\full", "$ResultsRoot\collection", "$ResultsRoot\accounting") | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File } | Where-Object { $_.Extension -in '.csv','.json' -and $_.FullName -notmatch '[\\/]checkpoints[\\/]' } | ForEach-Object { git add -- $_.FullName }
git add -- run_scripts/tuning/results/v3/full/selection_report.md
$staged = @(git diff --cached --name-only)
if ($staged | Where-Object { $_ -match 'checkpoints|\.pyc$|_wb/|\.(pt|pth|zip|tar|gz)$' }) { throw 'Model/archive/checkpoint/cache staged' }
git diff --cached --stat
```

**7.6 Commit and push.**
```powershell
git commit -m "Publish verified v3 tuning results and selected configurations"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin new_experiments
if ($LASTEXITCODE -ne 0) { throw 'Results push failed' }
```

## Part 8. Check that cloud cleanup is complete

**Destruction happens one machine at a time in Part 6, not here.** Do not keep a finished confirmation machine while waiting for Part 7 or for the other machines.

On Vast.ai, confirm these new confirmation instances have been destroyed:

| Confirmation machine | New instance ID | Its destruction step |
|---|---|---|
| A | `58344703` | **6A.6** |
| B | `58344678` | **6B.6** |
| C | `58344651` | **6C.6** |

If one is still listed, return to that machine's Part 6 steps. Once its own CSV/JSON and model downloads, local verification, external backups and any optional log copies are complete, destroy it. The other machines' progress is not a condition.

The original screening machines were destroyed separately in Part 3; their old IDs are not the confirmation IDs above.

## Part 9. When something goes wrong

### The log ends with `BATCH FAILED` or `PAUSED`, or a run was interrupted

`PAUSED` (and exit code 130 in `status.csv`) means someone pressed Ctrl+C in the training window. `BATCH FAILED` means at least one run failed; the other runs still ran.

1. Look at the error. Per-run output is in `results/v3/full/b/<machine>/<batch>/NN.out.log` and `NN.err.log`. A failed run's `status.json` holds the error message.
2. Start again with the matching **restart** command below, in the training tmux window. **Never re-run the plain start command.** The restart skips finished runs, continues an unfinished run from its last checkpoint (or starts it again if there is none), and keeps all old evidence.
3. If the same run fails again, stop and investigate. Keep all the files.

Machine A, screening restart:
```bash
set -o pipefail
source ~/tuning_a.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_a_tuning.sh --python ~/tcg/bin/python --stage screen --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_a.log
```

New confirmation machine A, confirmation restart (automatically reuses the time saved in 5A.3; keep the saved file unchanged):
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_a_tuning.sh --python ~/tcg/bin/python --stage confirm --continue-batch --wandb-mode online --billing-start "$(cat ~/tuning_a_confirm_start.txt)" 2>&1 | tee -a ~/tuning_a.log
```

Machine B, screening restart:
```bash
set -o pipefail
source ~/tuning_b.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_b_tuning.sh --python ~/tcg/bin/python --stage screen --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_b.log
```

New confirmation machine B, confirmation restart (automatically reuses the time saved in 5B.3; keep the saved file unchanged):
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_b_tuning.sh --python ~/tcg/bin/python --stage confirm --continue-batch --wandb-mode online --billing-start "$(cat ~/tuning_b_confirm_start.txt)" 2>&1 | tee -a ~/tuning_b.log
```

Machine C, screening restart:
```bash
set -o pipefail
source ~/tuning_c.env
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_c_tuning.sh --python ~/tcg/bin/python --stage screen --continue-batch --wandb-mode online --billing-start "$BILLING_START" --hourly-rate 0.30 --previously-billed-hours "$PREVIOUS_HOURS" 2>&1 | tee -a ~/tuning_c.log
```

New confirmation machine C, confirmation restart (automatically reuses the time saved in 5C.3; keep the saved file unchanged):
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/tuning/run_c_tuning.sh --python ~/tcg/bin/python --stage confirm --continue-batch --wandb-mode online --billing-start "$(cat ~/tuning_c_confirm_start.txt)" 2>&1 | tee -a ~/tuning_c.log
```

An interrupted run leaves a failed attempt folder next to the new one. That is expected. Leave it there; the checks use the completed attempt.

### `BUDGET WARNING`

The launcher never skips or stops a run because of the budget. Before a run that it expects to go past the machine's remaining hours, it prints `BUDGET WARNING ...` and continues. If you want to stop spending, stop the machine on Vast.ai yourself. Keep the saved start time unchanged; confirmation restart commands read it automatically.

### The SSH connection dropped

Training keeps running. Connect again with the same SSH command, then reattach tmux:
```bash
tmux attach
```

### Two completed attempts for the same run (duplicates)

The checks refuse to choose between them. Look at both attempts' identity, failure and resume files and decide which one counts. Keep both. Write a JSON object mapping the full config hash to the chosen relative `v3/full/.../aNN` folder, save it as `run_scripts/tuning/results/v3/full/choices.json`, and add `--attempt-choices run_scripts/tuning/results/v3/full/choices.json` to every launcher, check, collect and select command for that data. Only a verified complete attempt can be chosen. Keep the choices file when publishing.

Missing or failed screening runs block the finalist freeze. Going ahead with incomplete data needs a researcher decision and `--allow-incomplete --reason "..."`. Six valid confirmation runs per cell are always required.

### You need to re-pack a zip

`collect.py` refuses to overwrite an existing zip. First rename the old zips on that machine (nothing is deleted). You should see: no output and no error.
```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
for name in tuning_a_screen.zip tuning_a_confirm.zip tuning_b_screen.zip tuning_b_confirm.zip tuning_c_screen.zip tuning_c_confirm.zip; do
    if [ -f "$HOME/$name" ]; then mv -n "$HOME/$name" "$HOME/$name.$stamp"; fi
    if [ -f "$HOME/$name.sha256" ]; then mv -n "$HOME/$name.sha256" "$HOME/$name.sha256.$stamp"; fi
done
```
Then repeat the pack step. On the laptop, never overwrite a downloaded zip that differs: keep both. If the merge reports `Conflicting evidence`, keep both versions and work out which attempt is right.

## Part 10. Background: what runs and what it costs

Each cell tests 8 candidate settings (t00–t07) on seed **101** (screening). The top two per cell then train again on seeds **102, 103, 104** (confirmation). Which machine runs which slot is fixed in `trials_v3.json` and `finalists.json`.

Runs per machine (screening / confirmation):

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
| **Per stage** | **21/6** | **38/36** | **29/24** |
| **Total** | **27** | **74** | **53** |

Total: **88 screening + 66 confirmation = 154 runs**.

- **Training per run:** delivery 250,000; colour 750,000; maintenance 512,000 steps. That is 25% of the full budget, so late learners can be missed.
- **Evaluation:** at step 0 and every 5% (12,500 / 37,500 / 25,600 steps), with 50 / 50 / 20 episodes on seeds **3,000,000+i**. Final check: 200 / 200 / 100 episodes on seeds **3,010,000+i**.
- **Ranking:** greedy success AUC, then task progress AUC, then raw return AUC, then candidate ID. Confirmation uses the mean over the three new seeds. Noisy evaluation is reported but never used for ranking.
- **W&B:** project **new_experiments**, group **tuning-v3** (smoke and preflight: **tuning-v3-smoke**, **tuning-v3-preflight**).
- **Threads:** all runs use 1 CPU thread (the launcher default). The protocol planned 4. The first runs were at or under the planned time, so this was kept for consistency. Report it as a deviation.

Original spending estimates per role (historical screening plan; **no confirmation spending cap or carry-over**):

| Machine | Planned hours (linear estimate) | Expected billed hours | At $0.30/h | Hour share |
|---|---:|---:|---:|---:|
| A | 10.67 | 21–32 | $6.30–9.60 | 33.34 h |
| B | 10.64 | 21–32 | $6.30–9.60 | 33.33 h |
| C | 10.61 | 21–32 | $6.30–9.60 | 33.33 h |

The original plan targeted **100 billed hours / ~$30 compute** across both stages. That combined cap no longer applies: confirmation uses fresh rentals without carrying screening hours forward. Existing launcher warnings are advisory and do not stop training. Storage and transfer cost extra. Do not shorten training runs, reduce evaluations or drop cells to fit a spending estimate.

Measured first runs (2026-10-29): delivery DRQN ~7 min (plan 7.7), delivery DQN 4.4–5.0 min (plan 2.4), colour DQN 25 min (plan 14.7). All are within the launcher's 3× allowance.

**Machines:** RTX 3090 24 GB, at least 16 GB RAM. Rent at least 40 / 40 / 50 GB disk (A / B / C); preflight needs 20 / 20 / 25 GB free after installation. The original screening target was **$0.30/h or less**; confirmation has no spending cap. One training process per machine.

**Disk:** after each run is verified, its restart data is deleted to save space (`--prune-completed-restarts`, on by default). The final policy and all CSV/JSON evidence are kept. Unverified, failed or unfinished runs are never pruned. To keep restart data, add `--no-prune-completed-restarts` to both launcher and preflight, and provide 60 / 70 / 220 GB free instead.

**Screening billing record:** `billing_session.py --start` saved the screening rental start in `~/tuning_a.env` (or b/c). `--close` adds a one-hour allowance until stop/destruction. These screening files are not transferred to the fresh confirmation machines. For confirmation, the first command saves the machine's current UTC time in `~/tuning_a_confirm_start.txt` (or b/c); the launch and restart commands read it automatically. This marks the first setup command, not the earlier rental time. Generated accounting files are launcher bookkeeping and do not represent actual rental costs or total experiment spending.

| Result | Laptop | GitHub | W&B |
|---|---|---|---|
| CSV/JSON, trial list, finalists, selected configs, report | yes | after Part 7 | configs/metrics separately |
| Zips and `.sha256` files | Downloads + external backup | no | no |
| Greedy/noisy curves and summaries | CSV/JSON | CSV/JSON | tuning-v3 |
| Confirmation final policies + config/identity/metadata/summary | separate model archives in Downloads\tuning-confirmation-models + your external backup (Part 6.5) | no | no |
| Restart/replay data of finished runs | deleted on the machine | no | no |
| Restart/replay data of unfinished runs | on the machine until resolved | no | no |
| Per-run stdout/stderr and W&B cache | on the machine | no | W&B sync only |

## Part 11. Local validation and checks that only the cloud can do

T2: **107 tests**, **88/88** CPU/offline diagnostic runs (22 screening + 66 confirmation), 11 selections without override; all selected success AUCs zero. Four Windows checkpoint failures recovered; one failed attempt and three pre-resume statuses retained. Injected failure resumed from step 32. All **440 baseline hashes** unchanged; **55/55** baseline manifest. Original bundle: **1,286** CSV/JSON members, **735,954 bytes**, longest extracted path **223** characters. Thirty-one early metadata files lack a Git revision; this is engineering smoke evidence, not production science. See LOCAL_VALIDATION.md.

T3 results are in T3_VALIDATION.md. Only the cloud can test: package installation, CUDA/VRAM/real GPU updates, online W&B login and resume, native Linux interruption and continuation, SSH/scp, measured v3 speed and disk peaks, and billed cost. Each cloud preflight must pass before launch.

The laptop blocks marked for rehearsal (4.2, 4.3, 7.1, 7.2) can be rehearsed offline. You should see **`REHEARSAL PASSED`**, with 22/22 screening, 88/88 final and three identical finalist hashes. It uses a temporary folder under results and leaves production results untouched:

```powershell
& "$Repo\run_scripts\tuning\rehearse.ps1"
```

## Part 12. Collection contract for the stress-test phase

Root: run_scripts/tuning/results/v3/full. Attempts: env/algo/tNN/sNNN/aNN. All extracted/generated Windows paths must be below 240 characters. Downstream stress execution is separate.

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

Success and progress are defined by the task (deliveries, correct walls, services); return is the raw environment reward. The headline is greedy; noisy is secondary. Every objective comes from the identified attempt's own files, never the newest log folder. **Missing, failed, interrupted, invalid and nonfinite results are unavailable, never zero.** Check the exact budget, updates, evaluation records, seed independence and hashes before using any result. Policies and replay data are not in the CSV/JSON zips. Stress and final-comparison runs must use their own reserved seeds and protocol.
