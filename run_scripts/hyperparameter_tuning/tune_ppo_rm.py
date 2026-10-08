import os
# Prevent PyTorch OpenMP deadlock on Vast.ai high-core count instances
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import optuna
import wandb
import numpy as np
import torch
torch.set_num_threads(1)

import gymnasium as gym
import temporal_comp_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Setup paths for custom wrappers
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, "agents"))

from rm_wrapper import DeferredMaintenanceRMWrapper

class MaintenanceRewardMachine:
    def __init__(self, env):
        self.num_states = 2
    def get_initial_state(self):
        return 0
    def step(self, u, propositions):
        if u == -1: return -1, 0.0, True
        next_u = 1 if "is_ready" in propositions else u
        reward = sum(float(p.split("_")[1]) for p in propositions if p.startswith("reward_"))
        return next_u, reward, False

class PPORMObservationWrapper(gym.Wrapper):
    def __init__(self, env, rm_class):
        super().__init__(env)
        self.rm = rm_class(self.env)
        self.u = self.rm.get_initial_state()
        
        orig_shape = env.observation_space.shape[0]
        new_shape = orig_shape + self.rm.num_states
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )
        
    def _get_obs(self, obs):
        one_hot = np.zeros(self.rm.num_states, dtype=np.float32)
        if self.u != -1: 
            one_hot[self.u] = 1.0
        return np.concatenate([obs, one_hot])
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        propositions = info.get('propositions', set())
        
        next_u, rm_reward, rm_done = self.rm.step(self.u, propositions)
        self.u = next_u
        
        new_obs = self._get_obs(obs)
        final_done = terminated or rm_done
        return new_obs, rm_reward, final_done, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.u = self.rm.get_initial_state()
        return self._get_obs(obs), info

def make_env():
    env = gym.make("TemporalComp/DeferredMaintenance-v0")
    env = DeferredMaintenanceRMWrapper(env)
    env = PPORMObservationWrapper(env, MaintenanceRewardMachine)
    env = Monitor(env)
    return env

def objective(trial):
    # PPO Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.05, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    
    # Batch size must be a factor of n_steps
    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    env = make_env()
    eval_env = make_env()
    
    run = wandb.init(
        project="temporal-composition-gym",
        name=f"TemporalComp_DeferredMaintenance-v0__ppo_rm__{trial.number}__{int(time.time())}",
        config=trial.params,
        reinit=True
    )
    
    from wandb.integration.sb3 import WandbCallback
    from stable_baselines3.common.callbacks import CallbackList
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=1,
        device="cpu"
    )
    
    eval_callback = EvalCallback(
        eval_env, 
        eval_freq=50000,
        deterministic=True, 
        render=False,
        verbose=1
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=None,
        verbose=0
    )
    
    callbacks = CallbackList([eval_callback, wandb_callback])
    
    # Reduced timesteps for tuning (500k instead of 2M)
    model.learn(total_timesteps=500000, callback=callbacks)
    
    wandb.finish()
    
    # Return best mean reward
    if eval_callback.best_mean_reward == -np.inf:
        return -9999.0
        
    return eval_callback.best_mean_reward

if __name__ == "__main__":
    db_path = os.path.join(base_dir, "run_scripts", "hyperparameter_tuning", "ppo_rm_tuning.db")
    csv_path = os.path.join(base_dir, "run_scripts", "hyperparameter_tuning", "ppo_rm_tuning_results.csv")
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="ppo_rm_tuning",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )
    
    # Continuous control is CPU heavy, GPU can handle it since it's MLP policy. Safe to parallelize.
    study.optimize(objective, n_trials=20, n_jobs=1)
    
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
