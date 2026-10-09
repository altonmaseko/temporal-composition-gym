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

ALGORITHMS = ["PPO", "RecurrentPPO", "PPO-RM"]

def extract_success_rate(output):
    # Search for ep_rew_mean printed by stable_baselines3
    # Handles both normal floats and scientific notation (e.g., -1.12e+03)
    matches = re.findall(r"ep_rew_mean\s+\|\s+([-\d\.e\+]+)", output)
    if matches:
        return float(matches[-1])
    return 0.0

def main():
    wandb.init(
        project="temporal-composition-gym",
        name=f"stress_test_maintenance_{int(time.time())}",
    )
    
    results = []
    
    # Scale the temporal gap: (1, 2) to (8, 10)
    delay_ranges = [
        (1, 2), (2, 3), (3, 4), (4, 5), 
        (5, 6), (6, 7), (7, 8), (8, 10)
    ]
    
    for delay in delay_ranges:
        env_kwargs = json.dumps({"readiness_delay_range": delay})
        env_vars = os.environ.copy()
        env_vars["ENV_KWARGS"] = env_kwargs
        
        for algo in ALGORITHMS:
            print(f"\n=========================================")
            print(f"Running {algo} with delay range {delay}")
            print(f"=========================================\n")
            
            timesteps = 100000 # Benchmark baseline training timesteps
            
            if algo == "PPO":
                cmd = ["python", "agents/train_ppo.py", "--env", "TemporalComp/DeferredMaintenance-v0", "--timesteps", str(timesteps)]
            elif algo == "RecurrentPPO":
                cmd = ["python", "agents/train_recurrent_ppo.py", "--env", "TemporalComp/DeferredMaintenance-v0", "--timesteps", str(timesteps)]
            elif algo == "PPO-RM":
                # Use best hyperparameters from our Phase 3 tuning for PPO-RM
                cmd = [
                    "python", "agents/ppo_rm/train_ppo_rm.py", 
                    "--env-id", "TemporalComp/DeferredMaintenance-v0", 
                    "--timesteps", str(timesteps),
                    "--learning-rate", "0.00080869",
                    "--n-steps", "4096",
                    "--batch-size", "128",
                    "--clip-range", "0.2",
                    "--ent-coef", "0.025738"
                ]
                
            cmd[0] = sys.executable

            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env_vars, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            success_rate = extract_success_rate(result.stdout)
            print(f"Final score for {algo} (delay={delay}): {success_rate}")
            
            # Using delay max as x-axis metric for scaling temporal gap
            wandb.log({
                "difficulty_max_delay": delay[1],
                f"{algo}_success_rate": success_rate
            })
            
            results.append({
                "algorithm": algo,
                "delay_range": str(delay),
                "max_delay": delay[1],
                "success_rate": success_rate
            })

    os.makedirs("run_scripts/stress_testing/results", exist_ok=True)
    with open("run_scripts/stress_testing/results/maintenance_stress_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "delay_range", "max_delay", "success_rate"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
