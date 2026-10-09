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
        # Search for ep_rew_mean
        matches = re.findall(r"ep_rew_mean\s+\|\s+([-\d\.]+)", output)
        if matches:
            return float(matches[-1])
        return 0.0
    else:
        # Search for episodic_return
        matches = re.findall(r"episodic_return=([-\d\.]+)", output)
        if matches:
            return float(matches[-1])
        return 0.0

def main():
    wandb.init(
        project="temporal-composition-gym",
        name=f"stress_test_hazardous_{int(time.time())}",
    )

    results = []
    
    # Loop through the difficulty scale from 3 to 10
    for num_packages in range(3, 11):
        env_kwargs = json.dumps({"num_packages": num_packages})
        env_vars = os.environ.copy()
        env_vars["ENV_KWARGS"] = env_kwargs
        
        for algo in ALGORITHMS:
            print(f"\n=========================================")
            print(f"Running {algo} with {num_packages} packages")
            print(f"=========================================\n")
            
            timesteps = 100000 # Use 100k timesteps as baseline for training
            
            if algo == "DQN":
                cmd = ["python", "agents/train_dqn.py", "--env", "TemporalComp/HazardousDelivery-v0", "--timesteps", str(timesteps)]
            elif algo == "DRQN":
                p = HYPERPARAMS["DRQN"]
                cmd = ["python", "agents/drqn/train_drqn.py", "--env-id", "TemporalComp/HazardousDelivery-v0",
                       "--total-timesteps", str(timesteps),
                       "--batch-size", str(p["batch_size"]),
                       "--buffer-size", str(p["buffer_size"]),
                       "--exploration-fraction", str(p["exploration_fraction"]),
                       "--learning-rate", str(p["learning_rate"]),
                       "--seq-len", str(p["seq_len"])]
            elif algo == "QRM":
                p = HYPERPARAMS["QRM"]
                cmd = ["python", "agents/qrm/train_qrm.py", "--env-id", "TemporalComp/HazardousDelivery-v0",
                       "--total-timesteps", str(timesteps),
                       "--batch-size", str(p["batch_size"]),
                       "--buffer-size", str(p["buffer_size"]),
                       "--exploration-fraction", str(p["exploration_fraction"]),
                       "--learning-rate", str(p["learning_rate"]),
                       "--target-network-frequency", str(p["target_network_frequency"])]
            
            # Use same python executable as we're running with
            cmd[0] = sys.executable

            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env_vars, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            success_rate = extract_success_rate(algo, result.stdout)
            print(f"Final score for {algo} (packages={num_packages}): {success_rate}")
            
            wandb.log({
                "difficulty_num_packages": num_packages,
                f"{algo}_success_rate": success_rate
            })
            
            results.append({
                "algorithm": algo,
                "num_packages": num_packages,
                "success_rate": success_rate
            })

    os.makedirs("run_scripts/stress_testing/results", exist_ok=True)
    with open("run_scripts/stress_testing/results/hazardous_stress_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "num_packages", "success_rate"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
