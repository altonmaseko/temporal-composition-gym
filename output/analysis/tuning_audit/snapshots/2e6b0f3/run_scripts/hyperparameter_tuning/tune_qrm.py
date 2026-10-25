import os
import sys
import glob
import subprocess
import optuna
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_reward(run_dir):
    try:
        # Give it a tiny delay to ensure file buffers are flushed
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
    # Hyperparameters for QRM
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
    target_network_frequency = trial.suggest_categorical("target_network_frequency", [100, 500, 1000, 2000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.8)
    
    # Base command
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script_path = os.path.join(base_dir, "agents", "qrm", "train_qrm.py")
    
    # Note: Reduced timesteps for tuning (300k instead of 1M)
    cmd = [
        "python", script_path,
        "--env-id", "TemporalComp/HazardousDelivery-v0",
        "--total-timesteps", "300000",
        "--seed", str(trial.number),
        "--learning-rate", str(learning_rate),
        "--buffer-size", str(buffer_size),
        "--target-network-frequency", str(target_network_frequency),
        "--batch-size", str(batch_size),
        "--exploration-fraction", str(exploration_fraction)
    ]
    
    print(f"Trial {trial.number} starting...")
    # Run the script
    subprocess.run(cmd, cwd=base_dir, check=True)
    
    # Find the most recently created run folder in runs/
    runs_dir = os.path.join(base_dir, "runs")
    qrm_runs = glob.glob(os.path.join(runs_dir, "TemporalComp_HazardousDelivery-v0__qrm__*"))
    
    if not qrm_runs:
        return -9999.0
        
    latest_run = max(qrm_runs, key=os.path.getctime)
    print(f"Extracting reward from: {latest_run}")
    
    reward = get_latest_reward(latest_run)
    return reward

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path = os.path.join(base_dir, "run_scripts", "hyperparameter_tuning", "qrm_tuning.db")
    csv_path = os.path.join(base_dir, "run_scripts", "hyperparameter_tuning", "qrm_tuning_results.csv")
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="qrm_tuning",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )
    
    # Use n_jobs=8 for tabular environment as requested (CPU heavy)
    study.optimize(objective, n_trials=15, n_jobs=8)
    
    # Save results to CSV for easy viewing
    df = study.trials_dataframe()
    df.to_csv(csv_path)
    
    print(f"Results saved to {csv_path}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
