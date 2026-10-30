# Stress tests v3: production launch and collection

## How to use this guide

- **Bash** blocks run on a rented machine, in its SSH window. **PowerShell** blocks run on your laptop.
- Copy each block exactly. Run one block at a time. After each block, check the **You should see** line before going on.
- No script rents, stops or destroys a machine. You do that yourself on the Vast.ai website.
- In tmux, open a spare window with **Ctrl+b** then **c**, and go back with **Ctrl+b** then **0**. **Never press Ctrl+C in the training window**: it interrupts the run (if it happens, see Part 7).
- Replace every `<PLACEHOLDER>` (for example `<COMMIT>`, `<PORT_A>`, `<HOST_A>`) before running a block. The table in Part 1 is where you write them down.

What this phase runs (frozen in [STRESS_PROTOCOL.md](STRESS_PROTOCOL.md), decision D11): 154 runs, 11 cells × (5 seeds at the default level + 3 seeds at each of 3 other levels), at the full v3 budgets with the tuned configurations. Plan hash `098050611d58ce0fe2af4881794a115cb643dbc8b52d392378dde313b9e70d9c`.

## The order of work

| Step | Where | What | Part |
|---|---|---|---|
| 0 | Laptop | Commit and push the stress code, note `<COMMIT>` | 0 |
| 1 | Vast.ai | Rent three machines, fill in the table | 1 |
| 2 | Each machine | Set up, preflight, start the batch in tmux | 2 (A, B, C) |
| 3 | Each machine | Check in the first hour, then every 6–8 hours | 3 |
| 4 | Each machine, as soon as **it** prints **`BATCH COMPLETE`** | Check, collect and pack on the machine; download and verify results **and models** on the laptop; then **DESTROY that machine** | 4 (A, B, C) |
| 5 | Laptop, once **all three** are in | Final merge, 154/154 check, stage CSV/JSON only, stop for review | 5 |
| 6 | Vast.ai | Check nothing is left billing | 6 |
| — | Anywhere | Something went wrong | 7 |

The machines do not finish at the same time. **Finish and destroy each machine on its own, as soon as its own Part 4 steps pass, even while the others are still running.** Nothing is destroyed before its downloads verify. Only Part 5 waits for all three.

Expected running time (STRESS_PROTOCOL.md section 10; planning numbers from the tuning and v2 records, ±30% depending on the GPU):

| Machine | Runs | Expected | Order of work (each step is one "set": all 11 cells at one level and seed) |
|---|---:|---:|---|
| A | 54 | ~50 h | default s301 → easy s201 → default s304 → hard1 s202 → hard1 s203 (without its colour DRQN) |
| B | 54 | ~50 h | default s302 → hard1 s201 → default s305 → hard2 s202 → hard2 s203 (without its colour DRQN) |
| C | 46 | ~50 h | default s303 → hard2 s201 → easy s202 → easy s203 → colour DRQN hard1 s203 and hard2 s203 |

Partial-results milestones, roughly the same on all three machines: **~11 h** RQ1 has 3 seeds (default s301–s303); **~22 h** every level has one seed (a first RQ2 curve); **~32 h** RQ1 is complete (default s301–s305); **~50 h** everything. Inside each set the CPU cells (delivery, maintenance) run first and colour last, so colour (GPU) work starts about 3.5 h into each set.

## Part 0. Before renting: commit and push the stress code (laptop)

The stress code, plan and this guide are not committed yet. Ask me to commit and push them (or do it yourself), then read the commit hash. It becomes `<COMMIT>` everywhere below.

**0.1 Get `<COMMIT>`.** You should see: two identical 40-character hashes, then **`PUSHED`**. Copy that hash into the table in Part 1 as `<COMMIT>`. If the two hashes differ, the push has not happened yet: do not rent.
```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym'
git fetch origin new_experiments
$local = (git rev-parse HEAD).Trim(); $remote = (git rev-parse origin/new_experiments).Trim()
$local; $remote
if ($local -ne $remote) { throw 'Local commit is not pushed: push first' } else { 'PUSHED' }
```

## Part 1. Rent three machines

On Vast.ai, rent three machines with:

- **GPU:** RTX 3090 preferred. Faster cards (RTX 4070 Ti SUPER, 4080, 4090, A5000 or better) are fine. Every machine needs a GPU, because colour runs are spread across all three.
- **CUDA:** the offer must show **max CUDA ≥ 13.0** (tuning ran torch 2.14.0 with CUDA 13.0, driver 595).
- **Disk:** at least **60 GB**. **RAM:** at least **16 GB**, 32 GB preferred. **CPU:** 4 or more cores.
- **Image:** the Vast.ai **PyTorch (CUDA)** template on Ubuntu with Python 3.10–3.12 (tuning used Python 3.12.15). The setup below installs its own packages into `~/tcg`, so the template's own Python packages are not used.

Write the details here (keep this table; Part 6 uses it). Launch commit from Part 0: `<COMMIT>`.

| Role | Instance ID | GPU | SSH command from Vast.ai | Port | Host |
|---|---|---|---|---|---|
| A | `<INSTANCE_A>` | `<GPU_A>` | `ssh -p <PORT_A> root@<HOST_A> -L 8080:localhost:8080` | `<PORT_A>` | `<HOST_A>` |
| B | `<INSTANCE_B>` | `<GPU_B>` | `ssh -p <PORT_B> root@<HOST_B> -L 8080:localhost:8080` | `<PORT_B>` | `<HOST_B>` |
| C | `<INSTANCE_C>` | `<GPU_C>` | `ssh -p <PORT_C> root@<HOST_C> -L 8080:localhost:8080` | `<PORT_C>` | `<HOST_C>` |

If you are connected to more than one machine at once, the second and third SSH windows warn that port 8080 is in use. That is harmless.

## Part 2. Set up and start each machine

Do Part 2A, 2B and 2C. They are the same apart from the letter.

### Part 2A. Machine A

**2A.1 Connect** from laptop PowerShell with machine A's SSH command from the table.

**2A.2 Start tmux.** You should see: a green tmux status bar at the bottom.
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s stress_a; fi
```

**2A.3 Download the code at `<COMMIT>`.** You should see: the clone finish, then one commit line starting with the first characters of `<COMMIT>`, then **`CODE CHECK DONE`**. Anything else means stop.
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym && cd ~/temporal-composition-gym && git checkout --detach <COMMIT> && git log -1 --oneline && git diff --exit-code HEAD && echo "CODE CHECK DONE"
```

**2A.4 Install Python packages into `~/tcg`.** You should see: a Python version, then pip finishing without errors.
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```

**2A.5 Log in to W&B yourself.** You should see: a login success message.
```bash
~/tcg/bin/wandb login
```

**2A.6 Preflight.** It checks the interpreter and packages, the GPU, RAM (≥ 16 GB) and free disk (≥ 35 GB), W&B (it creates one online run in project `new_experiments`), rebuilds the stress plan **on this machine's CPU** and compares it field by field with the frozen file, checks `--threads 1`, and trains one colour cell for a few steps on CUDA. You should see: `[PASS] stress plan on this CPU: rebuilt plan equals the frozen file, hash 098050611d58...`, `[PASS] learning check: colour/dqn/default/s301 smoke on cuda: ...`, and last **`PREFLIGHT PASSED`**. `[WARN]` lines are fine. Fix every `[FAIL]` before starting (Part 7 covers a plan mismatch).
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/stress/preflight.py --role a --threads 1 --wandb-mode online 2>&1 | tee ~/preflight_a.log
```

**2A.7 Start the batch.** You should see: `role a: 54 runs, plan 098050611d58, mode full, threads 1`, then `ESTIMATE at launch: 50.1 h planned for this role`, then `START 1/54 delivery/dqn/...`. Each finished run prints `COMPLETE ... ESTIMATE: k/54 done, ~X h left`. Finished when the terminal prints **`BATCH COMPLETE`** (about 50 h later).
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/stress/run_a_stress.sh --python ~/tcg/bin/python --threads 1 --wandb-mode online 2>&1 | tee -a ~/stress_a.log
```

**2A.8 Detach and leave it running.** Press **Ctrl+b** then **d**. You should see: `[detached (from session stress_a)]`. You can now close SSH and turn off your laptop. To come back later, connect and run:
```bash
tmux attach -t stress_a
```

**2A.9 Check progress without touching the run.** Run this in a normal SSH shell (not inside the training window), or in a spare tmux window (**Ctrl+b** then **c**). You should see: the last log lines, the newest `progress.json` (`done`, `total`, `failures`, `remaining_h_estimate`) and the last finished runs with their `state` and `seconds`. Before the first run finishes, `progress.json` and `status.csv` do not exist yet ("No such file"); that is fine.
```bash
cd ~/temporal-composition-gym/run_scripts/stress/results/v3/full/b/a && B=$(ls -td */ | head -1) && tail -n 5 ~/stress_a.log; cat "$B/progress.json"; tail -n 5 "$B/status.csv"
```

### Part 2B. Machine B

**2B.1 Connect** from laptop PowerShell with machine B's SSH command from the table.

**2B.2 Start tmux.** You should see: a green tmux status bar at the bottom.
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s stress_b; fi
```

**2B.3 Download the code at `<COMMIT>`.** You should see: the clone finish, then one commit line starting with the first characters of `<COMMIT>`, then **`CODE CHECK DONE`**. Anything else means stop.
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym && cd ~/temporal-composition-gym && git checkout --detach <COMMIT> && git log -1 --oneline && git diff --exit-code HEAD && echo "CODE CHECK DONE"
```

**2B.4 Install Python packages into `~/tcg`.** You should see: a Python version, then pip finishing without errors.
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```

**2B.5 Log in to W&B yourself.** You should see: a login success message.
```bash
~/tcg/bin/wandb login
```

**2B.6 Preflight.** It checks the interpreter and packages, the GPU, RAM (≥ 16 GB) and free disk (≥ 35 GB), W&B (it creates one online run in project `new_experiments`), rebuilds the stress plan **on this machine's CPU** and compares it field by field with the frozen file, checks `--threads 1`, and trains one colour cell for a few steps on CUDA. You should see: `[PASS] stress plan on this CPU: rebuilt plan equals the frozen file, hash 098050611d58...`, `[PASS] learning check: colour/dqn/default/s302 smoke on cuda: ...`, and last **`PREFLIGHT PASSED`**. `[WARN]` lines are fine. Fix every `[FAIL]` before starting (Part 7 covers a plan mismatch).
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/stress/preflight.py --role b --threads 1 --wandb-mode online 2>&1 | tee ~/preflight_b.log
```

**2B.7 Start the batch.** You should see: `role b: 54 runs, plan 098050611d58, mode full, threads 1`, then `ESTIMATE at launch: 50.4 h planned for this role`, then `START 1/54 delivery/dqn/...`. Each finished run prints `COMPLETE ... ESTIMATE: k/54 done, ~X h left`. Finished when the terminal prints **`BATCH COMPLETE`** (about 50 h later).
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/stress/run_b_stress.sh --python ~/tcg/bin/python --threads 1 --wandb-mode online 2>&1 | tee -a ~/stress_b.log
```

**2B.8 Detach and leave it running.** Press **Ctrl+b** then **d**. You should see: `[detached (from session stress_b)]`. You can now close SSH and turn off your laptop. To come back later, connect and run:
```bash
tmux attach -t stress_b
```

**2B.9 Check progress without touching the run.** Run this in a normal SSH shell (not inside the training window), or in a spare tmux window (**Ctrl+b** then **c**). You should see: the last log lines, the newest `progress.json` (`done`, `total`, `failures`, `remaining_h_estimate`) and the last finished runs with their `state` and `seconds`. Before the first run finishes, `progress.json` and `status.csv` do not exist yet ("No such file"); that is fine.
```bash
cd ~/temporal-composition-gym/run_scripts/stress/results/v3/full/b/b && B=$(ls -td */ | head -1) && tail -n 5 ~/stress_b.log; cat "$B/progress.json"; tail -n 5 "$B/status.csv"
```

### Part 2C. Machine C

**2C.1 Connect** from laptop PowerShell with machine C's SSH command from the table.

**2C.2 Start tmux.** You should see: a green tmux status bar at the bottom.
```bash
if [ -z "${TMUX:-}" ]; then tmux new-session -A -s stress_c; fi
```

**2C.3 Download the code at `<COMMIT>`.** You should see: the clone finish, then one commit line starting with the first characters of `<COMMIT>`, then **`CODE CHECK DONE`**. Anything else means stop.
```bash
git clone --branch new_experiments https://github.com/altonmaseko/temporal-composition-gym.git ~/temporal-composition-gym && cd ~/temporal-composition-gym && git checkout --detach <COMMIT> && git log -1 --oneline && git diff --exit-code HEAD && echo "CODE CHECK DONE"
```

**2C.4 Install Python packages into `~/tcg`.** You should see: a Python version, then pip finishing without errors.
```bash
python3 -B --version && (python3 -B -m venv ~/tcg || (apt-get update && apt-get install -y python3-venv && python3 -B -m venv --clear ~/tcg))
```
```bash
~/tcg/bin/python -B -m pip install --upgrade pip && ~/tcg/bin/python -B -m pip install -r ~/temporal-composition-gym/requirements.txt
```

**2C.5 Log in to W&B yourself.** You should see: a login success message.
```bash
~/tcg/bin/wandb login
```

**2C.6 Preflight.** It checks the interpreter and packages, the GPU, RAM (≥ 16 GB) and free disk (≥ 35 GB), W&B (it creates one online run in project `new_experiments`), rebuilds the stress plan **on this machine's CPU** and compares it field by field with the frozen file, checks `--threads 1`, and trains one colour cell for a few steps on CUDA. You should see: `[PASS] stress plan on this CPU: rebuilt plan equals the frozen file, hash 098050611d58...`, `[PASS] learning check: colour/dqn/default/s303 smoke on cuda: ...`, and last **`PREFLIGHT PASSED`**. `[WARN]` lines are fine. Fix every `[FAIL]` before starting (Part 7 covers a plan mismatch).
```bash
set -o pipefail
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/stress/preflight.py --role c --threads 1 --wandb-mode online 2>&1 | tee ~/preflight_c.log
```

**2C.7 Start the batch.** You should see: `role c: 46 runs, plan 098050611d58, mode full, threads 1`, then `ESTIMATE at launch: 50.3 h planned for this role`, then `START 1/46 delivery/dqn/...`. Each finished run prints `COMPLETE ... ESTIMATE: k/46 done, ~X h left`. Finished when the terminal prints **`BATCH COMPLETE`** (about 50 h later).
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/stress/run_c_stress.sh --python ~/tcg/bin/python --threads 1 --wandb-mode online 2>&1 | tee -a ~/stress_c.log
```

**2C.8 Detach and leave it running.** Press **Ctrl+b** then **d**. You should see: `[detached (from session stress_c)]`. You can now close SSH and turn off your laptop. To come back later, connect and run:
```bash
tmux attach -t stress_c
```

**2C.9 Check progress without touching the run.** Run this in a normal SSH shell (not inside the training window), or in a spare tmux window (**Ctrl+b** then **c**). You should see: the last log lines, the newest `progress.json` (`done`, `total`, `failures`, `remaining_h_estimate`) and the last finished runs with their `state` and `seconds`. Before the first run finishes, `progress.json` and `status.csv` do not exist yet ("No such file"); that is fine.
```bash
cd ~/temporal-composition-gym/run_scripts/stress/results/v3/full/b/c && B=$(ls -td */ | head -1) && tail -n 5 ~/stress_c.log; cat "$B/progress.json"; tail -n 5 "$B/status.csv"
```

## Part 3. While the machines run

**In the first hour** on each machine:

1. The launch estimate (step 2X.7) should be **about 50 h** (A 50.1, B 50.4, C 50.3 planned). After the first finished runs, the `ESTIMATE:` line rescales by this machine's measured speed. A value between 35 and 65 h is normal; much more means a slow machine (see "The machine is far slower than planned" in Part 7).
2. The preflight log must show the colour learning check **on cuda**: `grep "learning check" ~/preflight_X.log` (X = a, b, c).
3. The first colour run of each machine starts about 3.5 h in. At your next check after that, confirm it is on the GPU. You should see `"selected": "cuda"` and your GPU name for every colour run that has started:
```bash
grep -h -E '"selected"|"gpu_name"' ~/temporal-composition-gym/run_scripts/stress/results/v3/full/colour/*/*/s*/a*/metadata.json | sort | uniq -c
```
While a colour run is active, `nvidia-smi` also lists one `python` process using about 1–2 GB of GPU memory.

**After that, check every 6–8 hours** with step 2X.9. What the last log line means:

- `START ...`: still running. Leave it.
- `COMPLETE ... ESTIMATE: ...`: a run finished. Leave it.
- **`BATCH COMPLETE`**: this machine is finished. Go to Part 4 for this machine.
- **`BATCH FAILED`** or **`PAUSED`**: something stopped or failed. Go to Part 7.

If `progress.json` shows `"failures"` above 0 while the batch is still running, one run failed and the batch moved on. Let it finish, then follow Part 7 ("BATCH FAILED").

Optional safety copy at a milestone (11 / 22 / 32 h): Part 4 step 4X.1 can be run any time to make a CSV/JSON bundle of what has finished so far, if you give `--out` a new name such as `~/stress_X_partial1.zip`. It does not disturb the running batch. Partial bundles are backups only; Part 4 still uses the final bundle.

## Part 4. When a machine prints **`BATCH COMPLETE`**

Do these steps for that machine right away. Don't wait for the other machines. **Download and verify the results and the models before destroying it.** Destroying the machine deletes its periodic policy snapshots, raw logs and W&B cache; only the final policy of each run is downloaded.

Before the first laptop block, open PowerShell and paste **4.0** (paste it again in any new PowerShell window).

**4.0 Laptop variables.**
```powershell
$ErrorActionPreference = 'Stop'
$Repo = 'C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym'
$Python = 'D:\miniconda3\envs\temporal_gym\python.exe'
$ResultsRoot = Join-Path $Repo 'run_scripts\stress\results'
$Mode = 'full'
$Bundles = Join-Path $HOME 'Downloads'
$ModelDownloads = [IO.Path]::GetFullPath((Join-Path $HOME 'Downloads\stress-models'))
$StageDryRun = $false
Set-Location -LiteralPath $Repo
```

### Part 4A. Machine A: check, collect, pack, download, verify, DESTROY

**4A.1 On machine A: check, collect and pack.** You should see, in order: **`54/54 complete (role a, mode full)`** with `unexpected 0`; then `... CSV/JSON files from role a bundled in /root/stress_a_full.zip; 54/54 runs verified complete; sha256 ...; no checkpoints or policies`; then **`MODELS PACKED: 54 final policies`** with a hash. If the manifest check fails, keep the machine and go to Part 7. Each output file refuses to overwrite an existing one; if you already made it, reuse it.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/stress/check_manifest.py --role a && ~/tcg/bin/python -B run_scripts/stress/collect.py --role a --out ~/stress_a_full.zip && ~/tcg/bin/python -B run_scripts/stress/pack_models.py pack --role a --out ~/stress_models_a_full.tar.gz
```

**4A.2 On laptop PowerShell: download the results bundle.** (Paste 4.0 first.) You should see: two files copied and no error.
```powershell
scp -P '<PORT_A>' 'root@<HOST_A>:stress_a_full.zip' 'root@<HOST_A>:stress_a_full.zip.sha256' $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Download failed: do not destroy machine A' }
```

**4A.3 On laptop PowerShell: verify the hash, merge, and check machine A on the laptop.** You should see **`HASH OK`**, **`BUNDLES VERIFIED AND MERGED`**, **`54/54 complete (role a, mode full)`** and **`MACHINE A RESULTS VERIFIED ON LAPTOP`**. Runs of other machines are not checked here.
<!-- rehearsal:verify-a -->
```powershell
$Zip = Join-Path $Bundles "stress_a_$Mode.zip"
$got = (Get-FileHash -LiteralPath $Zip).Hash.ToLower()
$want = (Get-Content -LiteralPath "$Zip.sha256").Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not destroy machine A' } else { "HASH OK $got" }
& "$Repo\run_scripts\stress\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @($Zip)
& $Python -B "$Repo\run_scripts\stress\check_manifest.py" --results-root $ResultsRoot --mode $Mode --role a
if ($LASTEXITCODE -ne 0) { throw 'Machine A incomplete on the laptop: do not destroy machine A' }
'MACHINE A RESULTS VERIFIED ON LAPTOP'
```

**4A.4 On laptop PowerShell: download the models OUTSIDE the repository.** This uses `$HOME\Downloads\stress-models`. You should see: the folder check pass silently, then two files copied. If the archive is already there, the block stops: keep the existing download and go straight to 4A.5.
```powershell
$Checkout = [IO.Path]::GetFullPath($Repo)
if ($ModelDownloads.StartsWith($Checkout, [StringComparison]::OrdinalIgnoreCase)) { throw 'Models must be downloaded outside the repository' }
New-Item -ItemType Directory -Path $ModelDownloads -Force | Out-Null
$ModelArchive = Join-Path $ModelDownloads "stress_models_a_$Mode.tar.gz"
if (Test-Path -LiteralPath $ModelArchive) { throw 'Model archive already downloaded: keep it and run 4A.5' }
scp -P '<PORT_A>' 'root@<HOST_A>:stress_models_a_full.tar.gz' 'root@<HOST_A>:stress_models_a_full.tar.gz.sha256' $ModelDownloads
if ($LASTEXITCODE -ne 0) { throw 'Model download failed: do not destroy machine A' }
```

**4A.5 On laptop PowerShell: verify the models.** This checks the archive against its `.sha256` file and every policy inside against the archive's own manifest. You should see **`MODELS VERIFIED: 54 final policies, role a, mode full, plan 098050611d58`** and **`MACHINE A MODELS VERIFIED`**.
<!-- rehearsal:models-a -->
```powershell
$ModelArchive = Join-Path $ModelDownloads "stress_models_a_$Mode.tar.gz"
if ($ModelArchive.StartsWith([IO.Path]::GetFullPath($Repo), [StringComparison]::OrdinalIgnoreCase)) { throw 'Models must stay outside the repository' }
& $Python -B "$Repo\run_scripts\stress\pack_models.py" verify --archive $ModelArchive
if ($LASTEXITCODE -ne 0) { throw 'MODEL HASH MISMATCH: do not destroy machine A' }
'MACHINE A MODELS VERIFIED'
```

**4A.6 NOW on Vast.ai: DESTROY machine A, instance `<INSTANCE_A>`.** Only after **`MACHINE A RESULTS VERIFIED ON LAPTOP`** and **`MACHINE A MODELS VERIFIED`** both appeared. Choose **Destroy** (not Stop: a stopped instance keeps billing for its disk). **Do this immediately, even if the other machines are still running.**

### Part 4B. Machine B: check, collect, pack, download, verify, DESTROY

**4B.1 On machine B: check, collect and pack.** You should see, in order: **`54/54 complete (role b, mode full)`** with `unexpected 0`; then `... CSV/JSON files from role b bundled in /root/stress_b_full.zip; 54/54 runs verified complete; sha256 ...; no checkpoints or policies`; then **`MODELS PACKED: 54 final policies`** with a hash. If the manifest check fails, keep the machine and go to Part 7. Each output file refuses to overwrite an existing one; if you already made it, reuse it.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/stress/check_manifest.py --role b && ~/tcg/bin/python -B run_scripts/stress/collect.py --role b --out ~/stress_b_full.zip && ~/tcg/bin/python -B run_scripts/stress/pack_models.py pack --role b --out ~/stress_models_b_full.tar.gz
```

**4B.2 On laptop PowerShell: download the results bundle.** (Paste 4.0 first.) You should see: two files copied and no error.
```powershell
scp -P '<PORT_B>' 'root@<HOST_B>:stress_b_full.zip' 'root@<HOST_B>:stress_b_full.zip.sha256' $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Download failed: do not destroy machine B' }
```

**4B.3 On laptop PowerShell: verify the hash, merge, and check machine B on the laptop.** You should see **`HASH OK`**, **`BUNDLES VERIFIED AND MERGED`**, **`54/54 complete (role b, mode full)`** and **`MACHINE B RESULTS VERIFIED ON LAPTOP`**. Runs of other machines are not checked here.
<!-- rehearsal:verify-b -->
```powershell
$Zip = Join-Path $Bundles "stress_b_$Mode.zip"
$got = (Get-FileHash -LiteralPath $Zip).Hash.ToLower()
$want = (Get-Content -LiteralPath "$Zip.sha256").Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not destroy machine B' } else { "HASH OK $got" }
& "$Repo\run_scripts\stress\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @($Zip)
& $Python -B "$Repo\run_scripts\stress\check_manifest.py" --results-root $ResultsRoot --mode $Mode --role b
if ($LASTEXITCODE -ne 0) { throw 'Machine B incomplete on the laptop: do not destroy machine B' }
'MACHINE B RESULTS VERIFIED ON LAPTOP'
```

**4B.4 On laptop PowerShell: download the models OUTSIDE the repository.** This uses `$HOME\Downloads\stress-models`. You should see: the folder check pass silently, then two files copied. If the archive is already there, the block stops: keep the existing download and go straight to 4B.5.
```powershell
$Checkout = [IO.Path]::GetFullPath($Repo)
if ($ModelDownloads.StartsWith($Checkout, [StringComparison]::OrdinalIgnoreCase)) { throw 'Models must be downloaded outside the repository' }
New-Item -ItemType Directory -Path $ModelDownloads -Force | Out-Null
$ModelArchive = Join-Path $ModelDownloads "stress_models_b_$Mode.tar.gz"
if (Test-Path -LiteralPath $ModelArchive) { throw 'Model archive already downloaded: keep it and run 4B.5' }
scp -P '<PORT_B>' 'root@<HOST_B>:stress_models_b_full.tar.gz' 'root@<HOST_B>:stress_models_b_full.tar.gz.sha256' $ModelDownloads
if ($LASTEXITCODE -ne 0) { throw 'Model download failed: do not destroy machine B' }
```

**4B.5 On laptop PowerShell: verify the models.** This checks the archive against its `.sha256` file and every policy inside against the archive's own manifest. You should see **`MODELS VERIFIED: 54 final policies, role b, mode full, plan 098050611d58`** and **`MACHINE B MODELS VERIFIED`**.
<!-- rehearsal:models-b -->
```powershell
$ModelArchive = Join-Path $ModelDownloads "stress_models_b_$Mode.tar.gz"
if ($ModelArchive.StartsWith([IO.Path]::GetFullPath($Repo), [StringComparison]::OrdinalIgnoreCase)) { throw 'Models must stay outside the repository' }
& $Python -B "$Repo\run_scripts\stress\pack_models.py" verify --archive $ModelArchive
if ($LASTEXITCODE -ne 0) { throw 'MODEL HASH MISMATCH: do not destroy machine B' }
'MACHINE B MODELS VERIFIED'
```

**4B.6 NOW on Vast.ai: DESTROY machine B, instance `<INSTANCE_B>`.** Only after **`MACHINE B RESULTS VERIFIED ON LAPTOP`** and **`MACHINE B MODELS VERIFIED`** both appeared. Choose **Destroy** (not Stop: a stopped instance keeps billing for its disk). **Do this immediately, even if the other machines are still running.**

### Part 4C. Machine C: check, collect, pack, download, verify, DESTROY

**4C.1 On machine C: check, collect and pack.** You should see, in order: **`46/46 complete (role c, mode full)`** with `unexpected 0`; then `... CSV/JSON files from role c bundled in /root/stress_c_full.zip; 46/46 runs verified complete; sha256 ...; no checkpoints or policies`; then **`MODELS PACKED: 46 final policies`** with a hash. If the manifest check fails, keep the machine and go to Part 7. Each output file refuses to overwrite an existing one; if you already made it, reuse it.
```bash
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/stress/check_manifest.py --role c && ~/tcg/bin/python -B run_scripts/stress/collect.py --role c --out ~/stress_c_full.zip && ~/tcg/bin/python -B run_scripts/stress/pack_models.py pack --role c --out ~/stress_models_c_full.tar.gz
```

**4C.2 On laptop PowerShell: download the results bundle.** (Paste 4.0 first.) You should see: two files copied and no error.
```powershell
scp -P '<PORT_C>' 'root@<HOST_C>:stress_c_full.zip' 'root@<HOST_C>:stress_c_full.zip.sha256' $Bundles
if ($LASTEXITCODE -ne 0) { throw 'Download failed: do not destroy machine C' }
```

**4C.3 On laptop PowerShell: verify the hash, merge, and check machine C on the laptop.** You should see **`HASH OK`**, **`BUNDLES VERIFIED AND MERGED`**, **`46/46 complete (role c, mode full)`** and **`MACHINE C RESULTS VERIFIED ON LAPTOP`**. Runs of other machines are not checked here.
<!-- rehearsal:verify-c -->
```powershell
$Zip = Join-Path $Bundles "stress_c_$Mode.zip"
$got = (Get-FileHash -LiteralPath $Zip).Hash.ToLower()
$want = (Get-Content -LiteralPath "$Zip.sha256").Split(' ')[0]
if ($got -ne $want) { throw 'HASH MISMATCH: do not destroy machine C' } else { "HASH OK $got" }
& "$Repo\run_scripts\stress\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @($Zip)
& $Python -B "$Repo\run_scripts\stress\check_manifest.py" --results-root $ResultsRoot --mode $Mode --role c
if ($LASTEXITCODE -ne 0) { throw 'Machine C incomplete on the laptop: do not destroy machine C' }
'MACHINE C RESULTS VERIFIED ON LAPTOP'
```

**4C.4 On laptop PowerShell: download the models OUTSIDE the repository.** This uses `$HOME\Downloads\stress-models`. You should see: the folder check pass silently, then two files copied. If the archive is already there, the block stops: keep the existing download and go straight to 4C.5.
```powershell
$Checkout = [IO.Path]::GetFullPath($Repo)
if ($ModelDownloads.StartsWith($Checkout, [StringComparison]::OrdinalIgnoreCase)) { throw 'Models must be downloaded outside the repository' }
New-Item -ItemType Directory -Path $ModelDownloads -Force | Out-Null
$ModelArchive = Join-Path $ModelDownloads "stress_models_c_$Mode.tar.gz"
if (Test-Path -LiteralPath $ModelArchive) { throw 'Model archive already downloaded: keep it and run 4C.5' }
scp -P '<PORT_C>' 'root@<HOST_C>:stress_models_c_full.tar.gz' 'root@<HOST_C>:stress_models_c_full.tar.gz.sha256' $ModelDownloads
if ($LASTEXITCODE -ne 0) { throw 'Model download failed: do not destroy machine C' }
```

**4C.5 On laptop PowerShell: verify the models.** This checks the archive against its `.sha256` file and every policy inside against the archive's own manifest. You should see **`MODELS VERIFIED: 46 final policies, role c, mode full, plan 098050611d58`** and **`MACHINE C MODELS VERIFIED`**.
<!-- rehearsal:models-c -->
```powershell
$ModelArchive = Join-Path $ModelDownloads "stress_models_c_$Mode.tar.gz"
if ($ModelArchive.StartsWith([IO.Path]::GetFullPath($Repo), [StringComparison]::OrdinalIgnoreCase)) { throw 'Models must stay outside the repository' }
& $Python -B "$Repo\run_scripts\stress\pack_models.py" verify --archive $ModelArchive
if ($LASTEXITCODE -ne 0) { throw 'MODEL HASH MISMATCH: do not destroy machine C' }
'MACHINE C MODELS VERIFIED'
```

**4C.6 NOW on Vast.ai: DESTROY machine C, instance `<INSTANCE_C>`.** Only after **`MACHINE C RESULTS VERIFIED ON LAPTOP`** and **`MACHINE C MODELS VERIFIED`** both appeared. Choose **Destroy** (not Stop: a stopped instance keeps billing for its disk). **Do this immediately, even if the other machines are still running.**

## Part 5. On the laptop, once all three machines are in

Paste 4.0 first if this is a new PowerShell window. Everything here runs on the laptop; no rented machine is needed.

**5.1 Recheck and merge all three bundles.** This rechecks every hash and accepts the identical files merged in Part 4. You should see **`BUNDLES VERIFIED AND MERGED: 3 bundle(s)`**.
<!-- rehearsal:final-merge -->
```powershell
& "$Repo\run_scripts\stress\merge_bundles.ps1" -ResultsRoot $ResultsRoot -Bundles @((Join-Path $Bundles "stress_c_$Mode.zip"), (Join-Path $Bundles "stress_a_$Mode.zip"), (Join-Path $Bundles "stress_b_$Mode.zip"))
```

**5.2 Check all 154 runs.** You should see **`154/154 complete (all runs, mode full)`** with roles A 54, B 54, C 46 and `unexpected 0`, then the per-level counts (default 55, easy 33, hard1 33, hard2 33), then **`ALL 154 RUNS VERIFIED`**.
<!-- rehearsal:final-check -->
```powershell
& $Python -B "$Repo\run_scripts\stress\check_manifest.py" --results-root $ResultsRoot --mode $Mode
if ($LASTEXITCODE -ne 0) { throw 'Final manifest incomplete' }
$m = (& $Python -B "$Repo\run_scripts\stress\check_manifest.py" --results-root $ResultsRoot --mode $Mode --json | ConvertFrom-Json)
$m.by_level | Format-List
if ($m.expected -ne 154 -or -not $m.all_complete) { throw 'Expected 154 verified runs' }
'ALL 154 RUNS VERIFIED'
```

**5.3 Check the model downloads.** `$HOME\Downloads\stress-models` holds `stress_models_a_full.tar.gz`, `..._b_...` and `..._c_...` with their `.sha256` files (about 1.2 GB together). They were verified in Part 4. They stay outside the repository and are never staged.

**5.4 Stage only the CSV/JSON results, then stop.** This stages the CSV/JSON files under `run_scripts/stress/results/v3/full`, by explicit path, and nothing else. You should see: the number of files, then a `git diff --cached --stat` list of `run_scripts/stress/results/v3/full/...` `.csv`/`.json` files only, then **`STAGED FOR REVIEW: do not commit yet`**. **Do not commit.** Send me the output so I can review it (Prompt 5 verifies the results before anything is committed).
<!-- rehearsal:stage -->
```powershell
Set-Location -LiteralPath $Repo
$files = @(Get-ChildItem -LiteralPath (Join-Path $ResultsRoot "v3\$Mode") -Recurse -File |
    Where-Object { $_.Extension -in '.csv', '.json' -and $_.FullName -notmatch '[\\/](checkpoints|_wb|logs|wandb)[\\/]' } |
    ForEach-Object { $_.FullName.Substring([IO.Path]::GetFullPath($Repo).TrimEnd('\').Length + 1).Replace('\', '/') })
"$($files.Count) CSV/JSON files selected"
$list = Join-Path $env:TEMP 'stress_stage_list.txt'
[IO.File]::WriteAllText($list, ($files -join "`n") + "`n")
if ($StageDryRun) { $out = @(git add --dry-run --pathspec-from-file=$list) } else { git restore --staged . ; git add --pathspec-from-file=$list; $out = @(git diff --cached --name-only) }
if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
$forbidden = 'checkpoints|_wb/|/logs/|\.pyc|\.(pt|zip|tar|gz)|results/r[0-9a-f]{4}'
if ($Mode -eq 'full') { $forbidden += '|/smoke/' }
if ($out | Where-Object { $_ -notmatch '\.(csv|json)''?$' -or $_ -match $forbidden }) { throw 'Something other than CSV/JSON results would be staged' }
if (-not $StageDryRun) { git diff --cached --stat | Select-Object -Last 3 }
"STAGED FOR REVIEW: do not commit yet ($($out.Count) entries)"
```

## Part 6. Final Vast.ai cleanup check

On the Vast.ai website:

1. **Instances:** `<INSTANCE_A>`, `<INSTANCE_B>` and `<INSTANCE_C>` are gone (destroyed in 4A.6, 4B.6, 4C.6), and no other instance is listed, including stopped ones. If one is still listed, go back to that machine's Part 4; destroy it once its own downloads have verified.
2. **Storage / volumes:** no volume is listed. A stopped instance or a detached volume keeps billing for disk.
3. **Billing:** the current hourly charge shows $0.00/h.

## Part 7. When something goes wrong

### The log ends with `BATCH FAILED` or `PAUSED`, or a run was interrupted

`PAUSED` (exit code 130 in `status.csv`) means the training window received Ctrl+C. `BATCH FAILED` means at least one run failed; the other runs still ran.

1. Find the failed runs: `grep -h ",failed," ~/temporal-composition-gym/run_scripts/stress/results/v3/full/b/X/*/status.csv` (X = a, b, c). Per-run output is in the same batch folder as `NNN.out.log` and `NNN.err.log` (NNN = slot), and the run's own `status.json` holds the error.
2. In the training tmux window, start again with that machine's **restart** command below. **Never re-run the plain start command** (it refuses to touch existing runs). The restart skips verified runs, resumes an unfinished run from its last restart checkpoint (every 10%), or starts it fresh if there is none, and keeps all old evidence.
3. If the same run fails again, stop and investigate. Keep all files and the machine.

Machine A restart:
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/stress/run_a_stress.sh --python ~/tcg/bin/python --threads 1 --wandb-mode online --continue-batch 2>&1 | tee -a ~/stress_a.log
```
Machine B restart:
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/stress/run_b_stress.sh --python ~/tcg/bin/python --threads 1 --wandb-mode online --continue-batch 2>&1 | tee -a ~/stress_b.log
```
Machine C restart:
```bash
set -o pipefail
cd ~/temporal-composition-gym && bash run_scripts/stress/run_c_stress.sh --python ~/tcg/bin/python --threads 1 --wandb-mode online --continue-batch 2>&1 | tee -a ~/stress_c.log
```

An interrupted run leaves its earlier attempt folder (`a00`) next to a new one only if it had no restart checkpoint; otherwise it continues in the same folder. Leave old attempts in place; the checks use the one verified attempt.

### The SSH connection dropped

Training keeps running inside tmux. Connect again with the same SSH command, then:
```bash
tmux attach -t stress_a
```
Use `stress_b` or `stress_c` on the other machines. If tmux says there is no session, the machine rebooted: check Vast.ai, then follow "A machine died" below.

### Preflight: the stress plan does not match on this CPU

Preflight prints `[FAIL] stress plan on this CPU: differs from the frozen file:` and the differing fields. Do not start the batch.

1. Check the code first: run 2X.3 again and make sure `CODE CHECK DONE` appears and `git log -1` shows `<COMMIT>`. Run `~/tcg/bin/python -B run_scripts/stress/generate_plan.py --check` to see every differing field.
2. The stress plan copies the tuned values verbatim and uses no random draws, so the tuning-phase NumPy problem should not occur. If the differing fields are floating-point values that differ only in the last digits (the tuning machine B symptom), use the NumPy fallback that was used on tuning machine B, in this shell only, and run preflight again:
```bash
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
cd ~/temporal-composition-gym && ~/tcg/bin/python -B run_scripts/stress/generate_plan.py --check
```
   If NumPy refuses a feature name in that list, remove that name and try again. If `--check` now passes, start the batch **from the same shell** (the variable must stay set for every run, including restarts; add the same `export` line before every restart command on this machine).
3. **This is a deviation and must be recorded:** write down the machine, the date, the exact `NPY_DISABLE_CPU_FEATURES` value and the fields that differed, and give them to me for the results report. It is also visible in each run's `metadata.json` only if you tell me, so do not skip the note.
4. If the differences are anything else (an integer, a string, a missing field), stop and send me the output. Do not start.

### A machine died (instance gone, or it will not come back)

The machine plan fixes the work of each role; runs are never moved between roles.

- **If the instance is only stopped** (Vast.ai shows it, disk intact): start it again, reconnect, `tmux new-session -A -s stress_X`, and run that machine's **restart** command. It continues where it stopped.
- **If the instance or its disk is gone:** rent a replacement with the same requirements (Part 1), record its details in the Part 1 table as the new machine for the **same role**, and do steps 2X.1–2X.6 for that role. Then start it with the **restart** command (with `--continue-batch`) and identical settings: same `<COMMIT>`, `--threads 1`, `--wandb-mode online`. Runs whose files were lost are run again from the start on the same seeds; nothing from the old machine is mixed in unless you merged its earlier bundle.
- **Record it:** date, role, old and new instance IDs and GPUs, the last `progress.json` you saw, and which runs were lost. Give the note to me for the results report (it is a deviation under the D11 replacement rule).

### A leftover W&B background process after a forced stop

After a run is killed, W&B's helper (`wandb-core`) can stay alive and keep files open. First make sure no run is active, then stop the helper. You should see: no `run_stress.py` line from the first command; then either nothing or the `wandb-core` lines; then nothing from the last command.
```bash
pgrep -af "run_stress.py|stress/launch.py" || echo "no run active"
pgrep -af wandb
pkill -f wandb-core; sleep 2; pgrep -af wandb-core
```
Only do this when the first line prints `no run active`. Offline W&B data that was never uploaded can be synced later with `~/tcg/bin/wandb sync <folder>` while the machine still exists.

### The machine is far slower than planned

If after several runs the `ESTIMATE:` line shows more than about 70 h left, check `nvidia-smi` (colour runs must be on the GPU), `free -g` (no swapping) and `nproc`. A slow GPU only lengthens wall-clock time; the science is unchanged. Decide whether to keep it or replace it (replacement: "A machine died").

## Part 8. Background

- **Runs per machine:** A 54, B 54, C 46. One process at a time. Colour runs use CUDA and fail rather than fall back to the CPU; delivery and maintenance run on the CPU with 1 thread (`--threads 1`, recorded in every `metadata.json`).
- **Seeds:** the default level uses training seeds 301–305 and evaluation episode seeds 5,000,000+ (final 5,010,000+); the other levels use training seeds 201–203 and 4,000,000+ (final 4,010,000+).
- **Storage on a machine:** restart data (up to about 9 GB during a colour run) is deleted after each run is independently verified. Policy snapshots (every 10%) stay on the machine until it is destroyed; only the final policy of each run is downloaded (about 1.2 GB for all 154 runs).
- **Results zip contract:** CSV/JSON only (same files as tuning plus the stress per-episode fields `required`, `progress_at_4000`, `return_at_4000`, `success_by_4000`), batch `status.csv` and `progress.json`, and one content-addressed `collection/<role>-<hash>.json` manifest per machine. **Missing, failed or interrupted runs are unavailable, never zero.**
- **Rehearsal:** `rehearse.ps1` runs the whole pipeline on the laptop in smoke mode (about an hour); `rehearse.ps1 -GuideOnly` runs this guide's laptop blocks marked `<!-- rehearsal:... -->` against an existing rehearsal's bundles and models.
