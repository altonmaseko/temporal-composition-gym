import os
import sys
import optuna
import numpy as np
import gymnasium as gym
import temporal_comp_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Setup paths for custom wrappers
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

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
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0,
        device="cuda"
    )
    
    eval_callback = EvalCallback(
        eval_env, 
        eval_freq=50000,
        deterministic=True, 
        render=False
    )
    
    # Reduced timesteps for tuning (500k instead of 2M)
    model.learn(total_timesteps=500000, callback=eval_callback)
    
    # Return best mean reward
    if eval_callback.best_mean_reward == -np.inf:
        return -9999.0
        
    return eval_callback.best_mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="ppo_rm_tuning")
    
    # Continuous control is CPU heavy, GPU can handle it since it's MLP policy. Safe to parallelize.
    study.optimize(objective, n_trials=20, n_jobs=4)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
