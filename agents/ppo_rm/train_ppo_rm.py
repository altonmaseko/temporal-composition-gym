import argparse
import os
import sys

import gymnasium as gym
from stable_baselines3 import PPO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.common.env_factory import ENV_IDS, make_env  # noqa: E402
from agents.reward_machines import RewardMachineWrapper, make_reward_machine  # noqa: E402

# Protocol v2: PPO-RM = SB3 PPO on the base maintenance observation plus the reward
# machine's task-state features [held codes (K), stage (S), ready]. The RM reward equals the
# env reward on every step (checked by the wrapper), so there is no reward shaping.

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
    parser.add_argument("--env-id", type=str, default="TemporalComp/DeferredMaintenance-v0")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    args = parser.parse_args()
    
    import time
    import wandb
    run_name = f"{args.env_id.replace('/', '_')}__ppo_rm__{args.seed}__{int(time.time())}"
    wandb.init(
        project="temporal-composition-gym",
        name=run_name,
        config=vars(args)
    )

    # Environment setup (protocol v2). Prompt 3 replaces checkpoints/W&B/evaluation below.
    env_key = {v: k for k, v in ENV_IDS.items()}[args.env_id]
    if env_key != "maintenance":
        raise ValueError("PPO-RM is the explicit baseline for Deferred Maintenance only")
    base = make_env(env_key)
    env = gym.wrappers.RecordEpisodeStatistics(base)
    env = RewardMachineWrapper(env, make_reward_machine(env_key, base), obs_mode="append", reward_mode="rm")

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
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
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
