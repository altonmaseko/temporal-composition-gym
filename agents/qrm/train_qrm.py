import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import temporal_comp_gym
from torch.utils.tensorboard import SummaryWriter

# Import our custom wrapper
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rm_wrapper import HazardousDeliveryRMWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="TemporalComp/HazardousDelivery-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-network-frequency", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    return parser.parse_args()

class DeliveryRewardMachine:
    """
    Finite State Automaton representing the logic of the Hazardous Delivery task.
    State u represents the number of packages delivered so far.
    """
    def __init__(self, env):
        self.num_packages = env.unwrapped.num_packages
        self.reward_delivery = env.unwrapped.reward_delivery
        self.penalty_damage = env.unwrapped.penalty_damage
        self.penalty_premature = env.unwrapped.penalty_premature_goal
        self.penalty_death = env.unwrapped.penalty_death_square
        self.reward_completion = env.unwrapped.reward_task_completion
        
        # Valid states: 0 to num_packages. -1 is a special terminal failure state.
        self.num_states = self.num_packages + 1
        
    def get_initial_state(self):
        return 0
        
    def step(self, u, propositions):
        """
        Simulates transitioning from RM state `u` given logical `propositions`.
        Returns (next_u, reward, done)
        """
        if u == -1: # Already in terminal state
            return -1, 0.0, True
            
        reward = 0.0
        done = False
        next_u = u
        
        # 1. Check catastrophic failures
        if 'died' in propositions or 'stepped_on_lava' in propositions:
            reward += self.penalty_death
            done = True
            next_u = -1
        elif 'reached_goal_prematurely' in propositions:
            reward += self.penalty_premature
            done = True
            next_u = -1
            
        # 2. Check incremental damage
        if 'stepped_on_damage' in propositions:
            reward += self.penalty_damage
            
        # 3. Check progress
        if not done:
            if 'delivered_package' in propositions:
                reward += self.reward_delivery
                next_u = min(u + 1, self.num_packages)
                
            if 'reached_goal_successfully' in propositions:
                reward += self.reward_completion
                done = True
                next_u = -1
                
        return next_u, reward, done

class QRMReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.ptr = 0
        
    def add(self, obs, action, next_obs, propositions, env_done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.ptr] = (obs, action, next_obs, propositions, env_done)
        self.ptr = (self.ptr + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class QRMNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions, num_rm_states):
        super().__init__()
        # MLPs for each RM state. This is an efficient way to represent Q(s, u, a).
        # We output a tensor of shape (batch, num_rm_states, num_actions)
        self.num_rm_states = num_rm_states
        self.num_actions = num_actions
        
        self.fc1 = nn.Linear(np.prod(obs_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_q = nn.Linear(64, num_rm_states * num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc_q(x)
        # Reshape to (batch, num_rm_states, num_actions)
        return q.view(-1, self.num_rm_states, self.num_actions)

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
    run_name = f"{args.env_id.replace('/', '_')}__qrm__{args.seed}__{int(time.time())}"
    
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
    
    # Environment Setup
    raw_env = gym.make(args.env_id)
    raw_env = gym.wrappers.RecordEpisodeStatistics(raw_env)
    env = HazardousDeliveryRMWrapper(raw_env)
    
    rm = DeliveryRewardMachine(env)
    
    q_network = QRMNetwork(env.observation_space.shape, env.action_space.n, rm.num_states).to(device)
    target_network = QRMNetwork(env.observation_space.shape, env.action_space.n, rm.num_states).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    rb = QRMReplayBuffer(args.buffer_size)
    
    obs, _ = env.reset(seed=args.seed)
    u = rm.get_initial_state()
    
    start_step = load_latest_checkpoint("models", args.env_id, "qrm", q_network, optimizer, args.seed)
    target_network.load_state_dict(q_network.state_dict())
    
    for global_step in range(start_step, args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = q_network(obs_t) # shape: (1, num_states, num_actions)
                action = torch.argmax(q_values[0, u]).item()
                
        next_obs, env_reward, terminated, truncated, info = env.step(action)
        env_done = terminated or truncated
        propositions = info['propositions']
        
        # Advance the active RM state
        next_u, rm_reward, rm_done = rm.step(u, propositions)
        
        rb.add(obs, action, next_obs, propositions, env_done)
        
        if env_done or rm_done:
            obs, _ = env.reset()
            u = rm.get_initial_state()
            if "episode" in info:
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        else:
            obs = next_obs
            u = next_u
            
        # Training Logic: Q-Learning for Reward Machines
        if len(rb.buffer) > args.batch_size:
            batch = rb.sample(args.batch_size)
            b_obs = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
            b_actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.long).to(device)
            b_next_obs = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(device)
            b_props = [b[3] for b in batch]
            b_env_dones = [b[4] for b in batch]
            
            # Forward pass for ALL RM states simultaneously
            q_values = q_network(b_obs) # (batch, num_states, num_actions)
            with torch.no_grad():
                target_q_values = target_network(b_next_obs)
                
            loss = 0
            # QRM core: Compute off-policy updates for *every* RM state u_i
            for u_i in range(rm.num_states):
                # Calculate what the reward and next state would have been if we were in u_i
                u_i_rewards = []
                u_i_next_states = []
                u_i_dones = []
                
                for i in range(args.batch_size):
                    nxt_u, rw, dn = rm.step(u_i, b_props[i])
                    # If the environment itself terminated, it overrides RM continuity
                    final_done = dn or b_env_dones[i]
                    u_i_rewards.append(rw)
                    u_i_next_states.append(nxt_u)
                    u_i_dones.append(final_done)
                    
                rw_t = torch.tensor(u_i_rewards, dtype=torch.float32).to(device)
                dn_t = torch.tensor(u_i_dones, dtype=torch.float32).to(device)
                
                # Gather Q-values for the actions taken
                q_u_i = q_values[:, u_i, :].gather(1, b_actions.unsqueeze(-1)).squeeze(-1)
                
                # Compute TD Target
                td_target = torch.zeros(args.batch_size, dtype=torch.float32).to(device)
                for i in range(args.batch_size):
                    if not dn_t[i]:
                        nxt_u = u_i_next_states[i]
                        if nxt_u != -1:
                            td_target[i] = rw_t[i] + args.gamma * torch.max(target_q_values[i, nxt_u])
                    else:
                        td_target[i] = rw_t[i]
                        
                loss += F.mse_loss(q_u_i, td_target)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if global_step % 500 == 0:
                writer.add_scalar("losses/qrm_loss", loss.item(), global_step)
                
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())
            
        if global_step > 0 and global_step % 100000 == 0:
            checkpoint_path = f"models/qrm_{args.env_id.split('/')[-1]}_seed{args.seed}_step_{global_step}.pt"
            torch.save({
                'global_step': global_step,
                'model_state_dict': q_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
    env.close()
    writer.close()
    
    os.makedirs("models", exist_ok=True)
    torch.save(q_network.state_dict(), f"models/qrm_{args.env_id.split('/')[-1]}_seed{args.seed}.pt")
