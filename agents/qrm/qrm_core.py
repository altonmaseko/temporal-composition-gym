"""Q-learning for Reward Machines (Toro Icarte et al. 2018) with protocol-v2 corrections.

One Q head per RM state (per abstract state phi(u) for QRM-Cue) on a shared encoder.
Every replayed transition updates every RM state for which the reward machine declares
the transition valid (rm.counterfactual), not only the state that was actually visited.
The rules for which counterfactual states are valid live with each reward machine.

Changes from the pilot implementation:
- Labels come from the env's info["events"]; RM rewards equal env rewards on real steps.
- Counterfactual targets are computed in one vectorised pass (no per-element GPU writes).
- Targets use the RM's terminal flag: time-limit truncations bootstrap.
- Replay is a FrameReplay sized in FRAMES (uint8 images stored once for QRM-Cue).
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.common.encoders import make_encoder
from agents.common.replay import FrameReplay


@dataclass
class QRMConfig:
    encoder: str = "mlp"
    enc_dim: int = 128
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 5e-4
    replay_frames: int = 100_000
    learning_starts: int = 10_000
    train_freq: int = 4
    target_update: int = 2_000
    batch_size: int = 32            # transitions per update (each updates every valid head)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.2
    grad_clip: float = 10.0

    def to_dict(self):
        return asdict(self)


QRM_DEFAULTS = {
    "delivery": QRMConfig(),
    "colour": QRMConfig(encoder="cnn", enc_dim=512, hidden_dim=256, lr=1e-4, learning_starts=20_000,
                        target_update=10_000),
}


class QRMNet(nn.Module):
    def __init__(self, obs_shape, n_actions, n_heads, cfg):
        super().__init__()
        self.encoder = make_encoder(obs_shape, cfg.encoder, cfg.enc_dim)
        self.body = nn.Sequential(nn.Linear(cfg.enc_dim, cfg.hidden_dim), nn.ReLU())
        self.out = nn.Linear(cfg.hidden_dim, n_heads * n_actions)
        self.n_heads, self.n_actions = n_heads, n_actions

    def forward(self, obs):
        z = self.body(self.encoder(obs))
        return self.out(z).view(-1, self.n_heads, self.n_actions)


def linear_epsilon(step, total_steps, cfg):
    frac = min(1.0, step / max(1.0, cfg.eps_fraction * total_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


class QRMAgent:
    EXTRA_KEYS = ("cls", "w", "c_next")

    def __init__(self, obs_space, n_actions, rm, cfg, device, rng):
        self.cfg = cfg
        self.rm = rm
        self.device = torch.device(device)
        self.n_actions = int(n_actions)
        self.n_heads = int(rm.n_states)
        self.rng = rng
        self.obs_shape = tuple(obs_space.shape)
        obs_dtype = np.uint8 if cfg.encoder == "cnn" else np.float32
        self.q = QRMNet(self.obs_shape, n_actions, self.n_heads, cfg).to(self.device)
        self.target = QRMNet(self.obs_shape, n_actions, self.n_heads, cfg).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        extras = {"head": ((), np.int16), "next_head": ((), np.int16)}
        extras.update({k: ((), np.int16) for k in self.EXTRA_KEYS})
        self.replay = FrameReplay(cfg.replay_frames, self.obs_shape, obs_dtype, extras=extras)
        self.n_updates = 0

    @torch.no_grad()
    def q_values(self, obs, head):
        x = torch.as_tensor(np.asarray(obs)[None], device=self.device)
        return self.q(x)[0, head].cpu().numpy()

    def act(self, obs, head, epsilon):
        if self.rng.exploration.random() < epsilon:
            return int(self.rng.exploration.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs, head)))

    def begin_episode(self, obs):
        self.replay.start_episode(obs)

    def observe(self, action, reward, terminated, truncated, next_obs, rm_info):
        ex = rm_info.get("extras", {})
        self.replay.add(action, rm_info["reward"], terminated, truncated, next_obs,
                        head=rm_info["head"], next_head=rm_info["next_head"],
                        **{k: ex.get(k, -1) for k in self.EXTRA_KEYS})

    def targets(self, batch):
        """Vectorised QRM targets; returns (q_sa, y, valid) tensors of shape (B, n_heads)."""
        dev = self.device
        valid, nxt, rew, term = self.rm.counterfactual(
            {k: batch[k] for k in self.EXTRA_KEYS}, batch["head"], batch["next_head"])
        obs = torch.as_tensor(batch["obs"], device=dev)
        next_obs = torch.as_tensor(batch["next_obs"], device=dev)
        act = torch.as_tensor(batch["action"], device=dev)
        q = self.q(obs)                                                     # (B, H, A)
        q_sa = q.gather(2, act[:, None, None].expand(-1, self.n_heads, 1)).squeeze(-1)
        with torch.no_grad():
            next_max = self.target(next_obs).max(-1).values                 # (B, H)
            nxt_t = torch.as_tensor(np.maximum(nxt, 0), device=dev, dtype=torch.int64)
            boot = next_max.gather(1, nxt_t)
            term_t = torch.as_tensor(term, device=dev, dtype=torch.float32)
            y = torch.as_tensor(rew, device=dev) + self.cfg.gamma * (1.0 - term_t) * boot
        return q_sa, y, torch.as_tensor(valid, device=dev, dtype=torch.float32)

    def compute_loss(self, batch):
        q_sa, y, valid = self.targets(batch)
        td = F.smooth_l1_loss(q_sa, y, reduction="none")
        loss = (td * valid).sum() / valid.sum().clamp(min=1.0)
        return loss, {"q_mean": float((q_sa.detach() * valid).sum() / valid.sum().clamp(min=1.0)),
                      "valid_heads_per_transition": float(valid.sum() / valid.shape[0])}

    def update(self):
        batch = self.replay.sample_transitions(self.cfg.batch_size, self.rng.replay)
        loss, info = self.compute_loss(batch)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.n_updates += 1
        info.update(loss=float(loss), grad_norm=float(grad_norm))
        return info

    def sync_target(self):
        self.target.load_state_dict(self.q.state_dict())


def train_qrm(env, agent, total_steps, reset_seed, on_episode=None, on_update=None):
    """Plain QRM loop; `env` must be wrapped in RewardMachineWrapper(obs_mode=None)."""
    cfg = agent.cfg
    obs, info = env.reset(seed=reset_seed)
    head = info["rm"]["head"]
    agent.begin_episode(obs)
    ep_return, ep_len = 0.0, 0
    for step in range(1, total_steps + 1):
        eps = linear_epsilon(step - 1, total_steps, cfg)
        action = agent.act(obs, head, eps)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.observe(action, reward, terminated, truncated, next_obs, info["rm"])
        ep_return += info.get("raw_reward", reward)
        ep_len += 1
        if terminated or truncated:
            if on_episode is not None:
                on_episode(step, ep_return, ep_len, info)
            obs, info = env.reset()
            head = info["rm"]["head"]
            agent.begin_episode(obs)
            ep_return, ep_len = 0.0, 0
        else:
            obs = next_obs
            head = info["rm"]["next_head"]
        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            metrics = agent.update()
            if on_update is not None:
                on_update(step, eps, metrics)
        if step % cfg.target_update == 0:
            agent.sync_target()
    return total_steps
