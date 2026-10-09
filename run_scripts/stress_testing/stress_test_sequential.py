import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

import json
import subprocess
import csv
import wandb
import time
import re
import sys

# Best hyperparameters from tuning
HYPERPARAMS = {
    "DRQN": {
        "batch_size": 16,
        "buffer_size": 10000,
        "exploration_fraction": 0.6433,
        "learning_rate": 1.2570e-05,
        "seq_len": 4
    },
    "QRM": {
        "batch_size": 32,
        "buffer_size": 100000,
        "exploration_fraction": 0.4093,
        "learning_rate": 2.2068e-05,
        "target_network_frequency": 2000
    }
}

ALGORITHMS = ["DQN", "DRQN", "QRM"]

def extract_success_rate(algo, output):
    if algo == "DQN":
        # Search for ep_rew_mean printed by stable_baselines3
        matches = re.findall(r"ep_rew_mean\s+\|\s+([-\d\.e\+]+)", output)
        if matches:
            return float(matches[-1])
        return 0.0
    else:
        # Search for episodic_return printed by cleanRL logs
        matches = re.findall(r"episodic_return=([-\d\.e\+]+)", output)
        if matches:
            return float(matches[-1])
        return 0.0

def main():
    wandb.init(
        project="temporal-composition-gym",
        name=f"stress_test_sequential_{int(time.time())}",
    )
    
    results = []
    
    # Scale the visual memory retention from 0 to 20
    delay_steps = [0, 2, 4, 6, 8, 10, 12, 16, 20]
    
    for delay in delay_steps:
        env_kwargs = json.dumps({"retention_delay_steps": delay})
        env_vars = os.environ.copy()
        env_vars["ENV_KWARGS"] = env_kwargs
        
        for algo in ALGORITHMS:
            print(f"\n=========================================")
            print(f"Running {algo} with retention delay {delay}")
            print(f"=========================================\n")
            
            timesteps = 100000 # Benchmark baseline training timesteps
            
            if algo == "DQN":
                cmd = ["python", "agents/train_dqn.py", "--env", "TemporalComp/SequentialColour-v0", "--timesteps", str(timesteps)]
            elif algo == "DRQN":
                p = HYPERPARAMS["DRQN"]
                cmd = ["python", "agents/drqn/train_drqn.py", "--env-id", "TemporalComp/SequentialColour-v0",
                       "--total-timesteps", str(timesteps),
                       "--batch-size", str(p["batch_size"]),
                       "--buffer-size", str(p["buffer_size"]),
                       "--exploration-fraction", str(p["exploration_fraction"]),
                       "--learning-rate", str(p["learning_rate"]),
                       "--seq-len", str(p["seq_len"])]
            elif algo == "QRM":
                p = HYPERPARAMS["QRM"]
                cmd = ["python", "agents/qrm/train_qrm.py", "--env-id", "TemporalComp/SequentialColour-v0",
                       "--total-timesteps", str(timesteps),
                       "--batch-size", str(p["batch_size"]),
                       "--buffer-size", str(p["buffer_size"]),
                       "--exploration-fraction", str(p["exploration_fraction"]),
                       "--learning-rate", str(p["learning_rate"]),
                       "--target-network-frequency", str(p["target_network_frequency"])]
                
            cmd[0] = sys.executable

            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env_vars, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            success_rate = extract_success_rate(algo, result.stdout)
            print(f"Final score for {algo} (delay={delay}): {success_rate}")
            
            wandb.log({
                "difficulty_retention_steps": delay,
                f"{algo}_success_rate": success_rate
            })
            
            results.append({
                "algorithm": algo,
                "retention_steps": delay,
                "success_rate": success_rate
            })

    os.makedirs("run_scripts/stress_testing/results", exist_ok=True)
    with open("run_scripts/stress_testing/results/sequential_stress_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "retention_steps", "success_rate"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
