import os
import sys
import glob
import subprocess
import optuna
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_reward(run_dir):
    try:
        import time
        time.sleep(2)
        event_acc = EventAccumulator(run_dir)
        event_acc.Reload()
        
        if 'charts/episodic_return' in event_acc.Tags().get('scalars', []):
            events = event_acc.Scalars('charts/episodic_return')
            if len(events) > 0:
                values = [e.value for e in events[-10:]]
                return np.mean(values)
    except Exception as e:
        print(f"Error reading tensorboard for {run_dir}: {e}")
    return -9999.0

def objective(trial):
    # Hyperparameters for DRQN
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [5000, 10000, 20000])
    seq_len = trial.suggest_categorical("seq_len", [4, 8, 16])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.8)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script_path = os.path.join(base_dir, "agents", "drqn", "train_drqn.py")
    
    # Reduced timesteps for tuning (300k instead of 3M)
    cmd = [
        "python", script_path,
        "--env-id", "TemporalComp/SequentialColour-v0",
        "--total-timesteps", "300000",
        "--seed", str(trial.number),
        "--learning-rate", str(learning_rate),
        "--buffer-size", str(buffer_size),
        "--seq-len", str(seq_len),
        "--batch-size", str(batch_size),
        "--exploration-fraction", str(exploration_fraction)
    ]
    
    print(f"Trial {trial.number} starting...")
    subprocess.run(cmd, cwd=base_dir, check=True)
    
    runs_dir = os.path.join(base_dir, "runs")
    drqn_runs = glob.glob(os.path.join(runs_dir, "TemporalComp_SequentialColour-v0__drqn__*"))
    
    if not drqn_runs:
        return -9999.0
        
    latest_run = max(drqn_runs, key=os.path.getctime)
    return get_latest_reward(latest_run)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="drqn_tuning")
    
    # Image-based environments can cause CUDA OOM, limit to n_jobs=2
    study.optimize(objective, n_trials=15, n_jobs=2)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
