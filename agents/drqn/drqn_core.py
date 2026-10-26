"""DRQN (Hausknecht & Stone 2015) with protocol-v2 corrections.

Changes from the pilot implementation:
- Visual encoder: NatureCNN on RGB frames (no flattening of pixels into an MLP, no grayscale).
- Replay: FrameReplay sized in FRAMES, uint8 images stored once (~2.1 GB at 100k frames).
- The recurrent state advances on every step, including epsilon-random actions.
- Training sequences never cross episodes; they start from the stored actor state of their
  first frame and include a burn-in context (R2D2, Kapturowski et al. 2019). A fraction of
  sequences is aligned to the episode start, where the zero state is exact.
- Padding and burn-in positions carry zero loss weight.
- Targets use (1 - terminated): time-limit truncations bootstrap from the final frame.
- Evaluation uses policy_step() with its own hidden state, so it never touches the
  training actor's state, replay or random streams.

Gradients flow through the burn-in frames as well (a longer truncated-BPTT window); the
loss itself is computed only on the trained positions.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.common.encoders import make_encoder
from agents.common.replay import FrameReplay


@dataclass
class DRQNConfig:
    encoder: str = "mlp"            # 'mlp' or 'cnn'
    enc_dim: int = 128
    lstm_dim: int = 128
    gamma: float = 0.99
    lr: float = 5e-4
    replay_frames: int = 100_000    # capacity unit: frames
    learning_starts: int = 10_000   # env steps before the first update
    train_freq: int = 4             # env steps per gradient update
    target_update: int = 2_000      # env steps between hard target syncs
    batch_size: int = 8             # sequences per update
    train_len: int = 32             # trained transitions per sequence
    burn_in: int = 16               # context frames before the trained part
    episode_start_fraction: float = 0.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.2       # fraction of the budget for the linear decay
    grad_clip: float = 10.0

    def to_dict(self):
        return asdict(self)


DRQN_DEFAULTS = {
    "delivery": DRQNConfig(),
    "colour": DRQNConfig(encoder="cnn", enc_dim=512, lstm_dim=256, lr=1e-4, learning_starts=20_000,
                         target_update=10_000, batch_size=4, train_len=96, burn_in=32,
                         episode_start_fraction=0.5),
}


class DRQNNet(nn.Module):
    def __init__(self, obs_shape, n_actions, cfg):
        super().__init__()
        self.encoder = make_encoder(obs_shape, cfg.encoder, cfg.enc_dim)
        self.lstm = nn.LSTM(cfg.enc_dim, cfg.lstm_dim, batch_first=True)
        self.head = nn.Linear(cfg.lstm_dim, n_actions)
        self.obs_ndim = len(obs_shape)

    def forward(self, obs_seq, hidden):
        """obs_seq (B, T, *obs_shape); hidden (h, c) each (1, B, lstm_dim)."""
        B, T = obs_seq.shape[:2]
        z = self.encoder(obs_seq.reshape((B * T,) + tuple(obs_seq.shape[2:]))).reshape(B, T, -1)
        out, hidden = self.lstm(z, hidden)
        return self.head(out), hidden


def linear_epsilon(step, total_steps, cfg):
    frac = min(1.0, step / max(1.0, cfg.eps_fraction * total_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


class DRQNAgent:
    def __init__(self, obs_space, n_actions, cfg, device, rng):
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_actions = int(n_actions)
        self.rng = rng
        self.obs_shape = tuple(obs_space.shape)
        self.obs_dtype = np.uint8 if cfg.encoder == "cnn" else np.float32
        self.q = DRQNNet(self.obs_shape, n_actions, cfg).to(self.device)
        self.target = DRQNNet(self.obs_shape, n_actions, cfg).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = FrameReplay(cfg.replay_frames, self.obs_shape, self.obs_dtype,
                                  hidden_shape=(2, cfg.lstm_dim))
        self.hidden = None
        self.n_updates = 0

    # ------------------------------------------------------------------ acting
    def zero_hidden(self, batch=1):
        z = torch.zeros(1, batch, self.cfg.lstm_dim, device=self.device)
        return (z, z.clone())

    def _hidden_np(self, hidden):
        return np.stack([hidden[0][0, 0].cpu().numpy(), hidden[1][0, 0].cpu().numpy()])

    @torch.no_grad()
    def policy_step(self, obs, hidden):
        """Pure function: Q-values for one observation and the next recurrent state."""
        x = torch.as_tensor(np.asarray(obs)[None, None], device=self.device)
        q, hidden = self.q(x, hidden)
        return q[0, 0].cpu().numpy(), hidden

    def begin_episode(self, obs):
        self.hidden = self.zero_hidden()
        self.replay.start_episode(obs, self._hidden_np(self.hidden))

    def act(self, obs, epsilon):
        q, self.hidden = self.policy_step(obs, self.hidden)   # always advance the state
        if self.rng.exploration.random() < epsilon:
            return int(self.rng.exploration.integers(self.n_actions))
        return int(np.argmax(q))

    def observe(self, action, reward, terminated, truncated, next_obs):
        self.replay.add(action, reward, terminated, truncated, next_obs,
                        next_hidden=self._hidden_np(self.hidden))

    # ------------------------------------------------------------------ learning
    def compute_loss(self, batch):
        q_sa, y, m = self.targets(batch)
        td = F.smooth_l1_loss(q_sa, y, reduction="none")
        loss = (td * m).sum() / m.sum().clamp(min=1.0)
        return loss, {"q_mean": float((q_sa.detach() * m).sum() / m.sum().clamp(min=1.0)),
                      "trained_transitions": float(m.sum())}

    def targets(self, batch):
        """(q_sa, y, mask), each (B, T-1): online Q of taken actions, TD targets, loss weights."""
        dev = self.device
        obs = torch.as_tensor(batch["obs"], device=dev)
        act = torch.as_tensor(batch["action"], device=dev)
        rew = torch.as_tensor(batch["reward"], device=dev)
        term = torch.as_tensor(batch["terminated"], device=dev, dtype=torch.float32)
        mask = torch.as_tensor(batch["loss_mask"], device=dev)
        h0 = torch.as_tensor(batch["h0"], device=dev)
        hidden = (h0[:, 0].unsqueeze(0).contiguous(), h0[:, 1].unsqueeze(0).contiguous())
        q, _ = self.q(obs, hidden)
        with torch.no_grad():
            qt, _ = self.target(obs, hidden)
            next_max = qt.max(-1).values
        q_sa = q[:, :-1].gather(2, act[:, :-1, None]).squeeze(-1)
        y = rew[:, :-1] + self.cfg.gamma * (1.0 - term[:, :-1]) * next_max[:, 1:]
        return q_sa, y, mask[:, :-1]

    def update(self):
        batch = self.replay.sample_sequences(self.cfg.batch_size, self.cfg.train_len, self.cfg.burn_in,
                                             self.rng.replay, self.cfg.episode_start_fraction)
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


def train_drqn(env, agent, total_steps, reset_seed, on_episode=None, on_update=None):
    """Plain training loop (logging, evaluation and checkpoints are added by the trainer).

    Returns the number of env steps executed (always total_steps).
    """
    cfg = agent.cfg
    obs, info = env.reset(seed=reset_seed)
    agent.begin_episode(obs)
    ep_return, ep_len = 0.0, 0
    for step in range(1, total_steps + 1):
        eps = linear_epsilon(step - 1, total_steps, cfg)
        action = agent.act(obs, eps)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.observe(action, reward, terminated, truncated, next_obs)
        ep_return += reward
        ep_len += 1
        if terminated or truncated:
            if on_episode is not None:
                on_episode(step, ep_return, ep_len, info)
            obs, info = env.reset()
            agent.begin_episode(obs)
            ep_return, ep_len = 0.0, 0
        else:
            obs = next_obs
        if step >= cfg.learning_starts and step % cfg.train_freq == 0 and agent.replay.num_transitions_added > 0:
            metrics = agent.update()
            if on_update is not None:
                on_update(step, eps, metrics)
        if step % cfg.target_update == 0:
            agent.sync_target()
    return total_steps
