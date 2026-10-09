import argparse
import os
import random
import time
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import temporal_comp_gym
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="TemporalComp/HazardousDelivery-v0",
        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size (in episodes)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the replay memory")
    parser.add_argument("--seq-len", type=int, default=8,
        help="the sequence length for backpropagation through time")
    parser.add_argument("--start-e", type=float, default=1.0,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    args = parser.parse_args()
    return args

class EpisodicReplayBuffer:
    def __init__(self, capacity, seq_len):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = []
        self.current_episode = []
        
    def add(self, obs, action, reward, next_obs, done):
        self.current_episode.append((obs, action, reward, next_obs, done))
        if done:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append(self.current_episode)
            self.current_episode = []
            
    def sample(self, batch_size):
        # Sample batch_size episodes
        sampled_episodes = random.sample(self.buffer, batch_size)
        
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        mask_batch = []
        
        for ep in sampled_episodes:
            ep_len = len(ep)
            if ep_len >= self.seq_len:
                start_idx = random.randint(0, ep_len - self.seq_len)
                slice_ep = ep[start_idx:start_idx + self.seq_len]
            else:
                slice_ep = ep
                
            obs, acts, rews, next_obs, dones = zip(*slice_ep)
            
            # Padding
            pad_len = self.seq_len - len(slice_ep)
            obs = list(obs) + [np.zeros_like(obs[0])] * pad_len
            next_obs = list(next_obs) + [np.zeros_like(next_obs[0])] * pad_len
            acts = list(acts) + [0] * pad_len
            rews = list(rews) + [0.0] * pad_len
            dones = list(dones) + [True] * pad_len
            mask = [1.0] * len(slice_ep) + [0.0] * pad_len
            
            obs_batch.append(obs)
            action_batch.append(acts)
            reward_batch.append(rews)
            next_obs_batch.append(next_obs)
            done_batch.append(dones)
            mask_batch.append(mask)
            
        return (
            torch.tensor(np.array(obs_batch), dtype=torch.float32),
            torch.tensor(np.array(action_batch), dtype=torch.long),
            torch.tensor(np.array(reward_batch), dtype=torch.float32),
            torch.tensor(np.array(next_obs_batch), dtype=torch.float32),
            torch.tensor(np.array(done_batch), dtype=torch.float32),
            torch.tensor(np.array(mask_batch), dtype=torch.float32)
        )

class DRQN(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.obs_shape = env.observation_space.shape
        self.num_obs_dims = len(self.obs_shape)
        obs_features = np.array(self.obs_shape).prod()
        self.fc1 = nn.Linear(obs_features, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc2 = nn.Linear(64, env.action_space.n)

    def forward(self, x, hidden):
        # x is either (1, *obs_shape) from env step or (batch, seq_len, *obs_shape) from buffer
        if len(x.shape) == self.num_obs_dims + 1:
            x = x.unsqueeze(1) # Add seq_len dimension: (batch, 1, *obs_shape)
            
        # Now x is (batch, seq_len, *obs_shape)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Flatten the observation dimensions
        x = x.reshape(batch_size, seq_len, -1)
            
        x = F.relu(self.fc1(x))
        out, hidden = self.lstm(x, hidden)
        q_values = self.fc2(out)
        return q_values, hidden
        
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

import glob
import re

def load_latest_checkpoint(models_dir, env_id, algo_name, model, optimizer, seed):
    os.makedirs(models_dir, exist_ok=True)
    env_name = env_id.split('/')[-1]
    search_pattern = os.path.join(models_dir, f"{algo_name}_{env_name}_seed{seed}_step_*.pt")
    checkpoints = glob.glob(search_pattern)
    
    if not checkpoints:
        return 0
        
    def extract_step(filepath):
        match = re.search(r'_step_(\d+)\.pt', filepath)
        return int(match.group(1)) if match else -1
        
    latest_ckpt = max(checkpoints, key=extract_step)
    print(f"Crash detected! Auto-resuming from: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['global_step']

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id.replace('/', '_')}__drqn__{args.seed}__{int(time.time())}"
    
    import wandb
    wandb.init(
        project="temporal-composition-gym",
        name=run_name,
        config=vars(args),
        sync_tensorboard=True
    )
    
    writer = SummaryWriter(f"runs/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import json
    import os
    env_kwargs = json.loads(os.environ.get("ENV_KWARGS", "{}"))
    env = gym.make(args.env_id, **env_kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    q_network = DRQN(env).to(device)
    target_network = DRQN(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    rb = EpisodicReplayBuffer(args.buffer_size, args.seq_len)
    
    obs, _ = env.reset(seed=args.seed)
    hidden = None
    
    start_step = load_latest_checkpoint("models", args.env_id, "drqn", q_network, optimizer, args.seed)
    target_network.load_state_dict(q_network.state_dict())
    
    for global_step in range(start_step, args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        # Action Logic
        if random.random() < epsilon:
            action = env.action_space.sample()
            hidden = None # Reset hidden state on random action to prevent bad history, or keep it (DRQN specific design choice)
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) # (1, features)
                q_values, hidden = q_network(obs_t, hidden)
                action = torch.argmax(q_values[0, -1]).item()
                
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        rb.add(obs, action, reward, next_obs, done)
        
        if done:
            obs, _ = env.reset()
            hidden = None
            if "episode" in info:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        else:
            obs = next_obs
            
        # Training Logic
        if len(rb.buffer) > args.batch_size:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones, b_masks = rb.sample(args.batch_size)
            b_obs = b_obs.to(device)
            b_actions = b_actions.to(device)
            b_rewards = b_rewards.to(device)
            b_next_obs = b_next_obs.to(device)
            b_dones = b_dones.to(device)
            b_masks = b_masks.to(device)
            
            # Forward pass
            q_values, _ = q_network(b_obs, None)
            # Gather q-values for the taken actions
            q_values = q_values.gather(2, b_actions.unsqueeze(-1)).squeeze(-1)
            
            with torch.no_grad():
                target_q_values, _ = target_network(b_next_obs, None)
                target_max = target_q_values.max(dim=2)[0]
                td_target = b_rewards + args.gamma * target_max * (1 - b_dones)
                
            loss = (F.mse_loss(q_values, td_target, reduction='none') * b_masks).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("losses/q_values", q_values.mean().item(), global_step)
                
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())
            
        if global_step > 0 and global_step % 100000 == 0:
            checkpoint_path = f"models/drqn_{args.env_id.split('/')[-1]}_seed{args.seed}_step_{global_step}.pt"
            torch.save({
                'global_step': global_step,
                'model_state_dict': q_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
    env.close()
    writer.close()
    
    os.makedirs("models", exist_ok=True)
    torch.save(q_network.state_dict(), f"models/drqn_{args.env_id.split('/')[-1]}_seed{args.seed}.pt")
