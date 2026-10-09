import argparse
import os
import numpy as np
import gymnasium as gym
import temporal_comp_gym
from stable_baselines3 import PPO

# Import our custom wrapper and RM logic
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rm_wrapper import HazardousDeliveryRMWrapper
from qrm.train_qrm import DeliveryRewardMachine

class PPORMObservationWrapper(gym.Wrapper):
    """
    Wraps an environment that emits 'propositions' (like HazardousDeliveryRMWrapper).
    It manages the Reward Machine state internally and appends it as a one-hot vector 
    to the agent's observation. This allows standard on-policy PPO to be RM-aware.
    """
    def __init__(self, env, rm_class):
        super().__init__(env)
        self.rm = rm_class(self.env)
        self.u = self.rm.get_initial_state()
        
        # Modify observation space to include the one-hot RM state
        orig_shape = env.observation_space.shape[0]
        new_shape = orig_shape + self.rm.num_states
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )
        
    def _get_obs(self, obs):
        one_hot = np.zeros(self.rm.num_states, dtype=np.float32)
        if self.u != -1: # -1 is terminal state
            one_hot[self.u] = 1.0
        return np.concatenate([obs, one_hot])
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        propositions = info.get('propositions', set())
        
        # Step the Reward Machine
        next_u, rm_reward, rm_done = self.rm.step(self.u, propositions)
        self.u = next_u
        
        # PPO-RM uses RM rewards instead of raw environment rewards!
        new_obs = self._get_obs(obs)
        final_done = terminated or rm_done
        
        return new_obs, rm_reward, final_done, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.u = self.rm.get_initial_state()
        return self._get_obs(obs), info

import glob
import re
from stable_baselines3.common.callbacks import CheckpointCallback

def load_latest_checkpoint_sb3(models_dir, env_id, algo_name):
    os.makedirs(models_dir, exist_ok=True)
    env_name = env_id.split('/')[-1]
    search_pattern = os.path.join(models_dir, f"{algo_name}_{env_name}_step_*_steps.zip")
    checkpoints = glob.glob(search_pattern)
    
    if not checkpoints:
        return None, 0
        
    def extract_step(filepath):
        match = re.search(r'_step_(\d+)_steps\.zip', filepath)
        return int(match.group(1)) if match else -1
        
    latest_ckpt = max(checkpoints, key=extract_step)
    print(f"Crash detected! Auto-resuming from: {latest_ckpt}")
    return latest_ckpt, extract_step(latest_ckpt)


def main():
    parser = argparse.ArgumentParser(description="Train PPO-RM on Temporal Composition Gym")
    parser.add_argument("--env-id", type=str, default="TemporalComp/HazardousDelivery-v0")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.00080869)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.025738)
    args = parser.parse_args()
    
    import time
    import wandb
    run_name = f"{args.env_id.replace('/', '_')}__ppo_rm__{args.seed}__{int(time.time())}"
    wandb.init(
        project="temporal-composition-gym",
        name=run_name,
        config=vars(args)
    )

    # Environment Setup
    # 1. Base Environment
    import json
    env_kwargs = json.loads(os.environ.get("ENV_KWARGS", "{}"))
    env = gym.make(args.env_id, **env_kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    # 2. Add Stats Tracker
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # 3. Add RM Proposition Emitter & State Augmentation
    if "HazardousDelivery" in args.env_id:
        env = HazardousDeliveryRMWrapper(env)
        env = PPORMObservationWrapper(env, DeliveryRewardMachine)
    elif "DeferredMaintenance" in args.env_id:
        from rm_wrapper import DeferredMaintenanceRMWrapper
        env = DeferredMaintenanceRMWrapper(env)
        
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
                
        env = PPORMObservationWrapper(env, MaintenanceRewardMachine)
    else:
        raise ValueError(f"Unsupported env_id for PPO-RM: {args.env_id}")

    # Initialize standard Stable Baselines 3 PPO!
    # Because the RM state is now part of the observation, PPO automatically learns temporal composition.
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./models/',
        name_prefix=f'ppo_rm_{args.env_id.split("/")[-1]}_step'
    )

    latest_ckpt, start_step = load_latest_checkpoint_sb3("models", args.env_id, "ppo_rm")
    
    from wandb.integration.sb3 import WandbCallback
    wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run_name}")
    
    if latest_ckpt:
        print(f"Loading model from {latest_ckpt}...")
        model = PPO.load(latest_ckpt, env=env, tensorboard_log=f"./runs/ppo_rm_{args.env_id.replace('/', '_')}")
        remaining_timesteps = args.timesteps - start_step
        if remaining_timesteps > 0:
            print(f"Resuming PPO-RM training for {remaining_timesteps} timesteps...")
            model.learn(total_timesteps=remaining_timesteps, reset_num_timesteps=False, callback=[checkpoint_callback, wandb_callback])
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate if hasattr(args, 'learning_rate') else 0.00080869,
            n_steps=args.n_steps if hasattr(args, 'n_steps') else 4096,
            batch_size=args.batch_size if hasattr(args, 'batch_size') else 128,
            clip_range=args.clip_range if hasattr(args, 'clip_range') else 0.2,
            ent_coef=args.ent_coef if hasattr(args, 'ent_coef') else 0.025738,
            seed=args.seed,
            verbose=1,
            tensorboard_log=f"./runs/ppo_rm_{args.env_id.replace('/', '_')}"
        )
        print(f"Starting PPO-RM training on {args.env_id} for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps, callback=[checkpoint_callback, wandb_callback])
    
    os.makedirs("models", exist_ok=True)
    save_path = f"models/ppo_rm_{args.env_id.split('/')[-1]}_seed{args.seed}"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    env.close()

if __name__ == "__main__":
    main()
