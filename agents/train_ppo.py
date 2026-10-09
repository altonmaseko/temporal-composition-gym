import argparse
import os
import gymnasium as gym
import temporal_comp_gym
from stable_baselines3 import PPO

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
    parser = argparse.ArgumentParser(description="Train PPO on Temporal Composition Gym")
    parser.add_argument("--env", type=str, required=True, 
                        choices=[
                            "TemporalComp/HazardousDelivery-v0", 
                            "TemporalComp/DeferredMaintenance-v0",
                            "TemporalComp/SequentialColour-v0"
                        ],
                        help="Environment ID")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    import time
    import wandb
    run_name = f"{args.env.replace('/', '_')}__ppo__{args.seed}__{int(time.time())}"
    wandb.init(
        project="temporal-composition-gym",
        name=run_name,
        config=vars(args)
    )

    # Environment-specific configurations
    config = {
        "TemporalComp/HazardousDelivery-v0": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64
        },
        "TemporalComp/DeferredMaintenance-v0": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64
        },
        "TemporalComp/SequentialColour-v0": {
            "policy": "CnnPolicy",
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 128
        }
    }

    env_settings = config[args.env]
    import json
    env_kwargs = json.loads(os.environ.get("ENV_KWARGS", "{}"))
    env = gym.make(args.env, **env_kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    
    # Wrapper for tracking episode stats
    env = gym.wrappers.RecordEpisodeStatistics(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./models/',
        name_prefix=f'ppo_{args.env.split("/")[-1]}_step'
    )

    latest_ckpt, start_step = load_latest_checkpoint_sb3("models", args.env, "ppo")
    
    from wandb.integration.sb3 import WandbCallback
    wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run_name}")
    
    if latest_ckpt:
        print(f"Loading model from {latest_ckpt}...")
        model = PPO.load(latest_ckpt, env=env, tensorboard_log=f"./runs/ppo_{args.env.replace('/', '_')}")
        remaining_timesteps = args.timesteps - start_step
        if remaining_timesteps > 0:
            print(f"Resuming PPO training for {remaining_timesteps} timesteps...")
            model.learn(total_timesteps=remaining_timesteps, reset_num_timesteps=False, callback=[checkpoint_callback, wandb_callback])
    else:
        model = PPO(
            policy=env_settings["policy"],
            env=env,
            learning_rate=env_settings["learning_rate"],
            n_steps=env_settings["n_steps"],
            batch_size=env_settings["batch_size"],
            seed=args.seed,
            verbose=1,
            tensorboard_log=f"./runs/ppo_{args.env.replace('/', '_')}"
        )
        print(f"Starting PPO training on {args.env} for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps, callback=[checkpoint_callback, wandb_callback])
    
    os.makedirs("models", exist_ok=True)
    save_path = f"models/ppo_{args.env.split('/')[-1]}_seed{args.seed}"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    env.close()

if __name__ == "__main__":
    main()
