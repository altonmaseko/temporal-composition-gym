# Phase 3: Baseline Evaluation Execution Guide 🚀
*(Comprehensive Manual for Temporal Composition Gym)*

Welcome to the Phase 3 Execution Manual. This document is written to be entirely foolproof. Even if someone has no context about your code, they can follow these steps from top to bottom and successfully generate all the data required for Phase 3 of your research proposal (Sample Efficiency & Temporal Degradation baselines).

---

## 🗂️ Overview of What You Will Execute
Your research compares three classes of algorithms across three environments. To make this easy, the execution commands have been permanently saved into three master bash scripts located in the `run_scripts/` folder:

1. **`run_memoryless_baselines.sh`**: Runs DQN and PPO (The Control Group).
2. **`run_implicit_memory_baselines.sh`**: Runs DRQN and Recurrent PPO (Sequence Models).
3. **`run_explicit_memory_baselines.sh`**: Runs QRM and PPO-RM (Automata Models).

Each script automatically loops through **3 random seeds (1, 2, 3)** for statistical validity and points each algorithm to its correct environment.

---

## STEP 1: Renting & Preparing the Vast.ai Server

Reinforcement Learning requires massive System RAM (for Replay Buffers) and strong GPUs (for Convolutional Neural Networks).

1. Go to [Vast.ai](https://vast.ai) (ensure you have at least $10-$20 in billing).
2. Click **Create Instance**.
3. **Template:** Select the official **PyTorch** template (e.g., PyTorch 2.x).
4. **Filters (CRITICAL):**
   - **GPU:** RTX 3090 or RTX 4090 (You need 24GB VRAM for the Image-based DRQN).
   - **RAM:** Minimum **64 GB** (Filter for `RAM >= 64`).
   - **Verification:** Check the **"Verified"** toggle to ensure the data center won't randomly shut off.
5. Hit **Rent**, wait 2-3 minutes for the server to boot up, and click **Connect**.
6. Copy the SSH command provided. It will look something like:
   `ssh -p 12345 root@192.168.x.x -L 8080:localhost:8080`

---

## STEP 2: Bootstrapping the Server Code

Open your local laptop's terminal (PowerShell or Command Prompt) and paste your SSH command to connect to the Vast.ai server. Once connected, type the following commands exactly as shown:

**1. Clone your project to the server:**
*(If your code is on GitHub. If it's strictly local, you will need to use `scp` or a tool like WinSCP/Cyberduck to drag and drop your `temporal-composition-gym` folder onto the server).*
```bash
git clone https://github.com/your-username/temporal-composition-gym.git
cd temporal-composition-gym
```

**2. Install all required dependencies:**
This installs Gymnasium, PyTorch, Stable Baselines 3, and WandB.
```bash
pip install -r requirements.txt
```

**3. Authenticate with WandB (Your Data Dashboard):**
This links the remote server to your personal web dashboard so you can view the charts live.
```bash
wandb login
```
*It will ask for an API key. Go to [wandb.ai](https://wandb.ai) -> Settings -> API Keys, copy it, and paste it into the terminal. (Note: the terminal will not show the characters as you paste them. Just press Enter).*

---

## STEP 3: Executing the Baseline Scripts (The Right Way)

**🚨 CRITICAL WARNING:** If you simply run a Python script in your SSH terminal, the script will instantly be killed the second your laptop goes to sleep or your Wi-Fi flickers. You MUST use a tool called `tmux` to run things in the background.

**1. Start a background terminal session:**
```bash
tmux new -s phase3
```
*You will notice a green bar appear at the bottom of your screen. This means you are safely inside the virtual session.*

**2. Make the Bash scripts executable:**
Before Linux will let you run the scripts in the `run_scripts/` folder, you must grant them permission.
```bash
chmod +x run_scripts/*.sh
```

**3. Run the scripts (One by one):**
Execute the first script. This will automatically train the agents, handle the environments, and save the checkpoints.
```bash
./run_scripts/run_memoryless_baselines.sh
```

**4. Detach and go to sleep:**
While the script is running, press **`Ctrl + B`**, let go of both keys, and then press **`D`**.
*You will be kicked out to your normal terminal. The script is now safely running in the background! You can close your laptop.*

**5. Check on the progress later:**
When you wake up, SSH back into your Vast.ai server and type:
```bash
tmux attach -t phase3
```
*This drops you right back into the screen where the script is running. Once Script 1 finishes, run Script 2, and then Script 3.*

---

## STEP 4: Handling "Bad Scenarios" (Crash Protocol)

### Scenario A: The Vast.ai Server Loses Power / Reboots
If the physical data center crashes, Vast.ai will reboot your machine, but your scripts will stop.
**The Fix:** 
Because we added `load_latest_checkpoint` to every Python file, you just need to SSH back in, open a new `tmux` session, and re-run the `.sh` script (e.g., `./run_scripts/run_memoryless_baselines.sh`). The Python files will automatically scan the `models/` folder, print `"Crash detected! Auto-resuming..."`, and pick up exactly where they died.

### Scenario B: "Killed" (System Out of Memory)
If the terminal prints `Killed` and stops, your Replay Buffer used up all 64GB of System RAM.
**The Fix:** 
Open `run_scripts/run_implicit_memory_baselines.sh` and change the `--total-timesteps` for DRQN Image tasks, or open the python files and lower the `buffer_size` arguments.

### Scenario C: CUDA Out of Memory
If you get `RuntimeError: CUDA out of memory`.
**The Fix:** Your batch size is too big for the GPU. Lower the `--batch-size` parameter in the corresponding Python file (e.g., from 64 to 32).

---

## STEP 5: Exporting Data for Phase 5 (Synthesis)

You never need to look at terminal logs to get your data.
1. Open your web browser on your laptop and go to **wandb.ai**.
2. Open your `temporal-composition-gym` project.
3. You will see beautiful, live-updating charts for all 6 algorithms across all 3 seeds.
4. Hover over any chart (like `rollout/ep_rew_mean` or Expected Cumulative Reward).
5. Click the **three vertical dots** (Options) -> **Download as CSV**.
6. You now have the exact data required to plot the Area Under the Curve (AUC) for your Phase 5 final research report!
