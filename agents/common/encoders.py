"""Observation encoders shared by DQN-family agents (protocol v2 section 6.2).

Images stay RGB (no grayscale: the colour task is decided by hue) and are only rescaled
to [0, 1]. Frames are 84x84 already, so no resizing is needed.
"""
import numpy as np
import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """Mnih et al. (2015) encoder. Input: uint8 (B, H, W, C) or (B, C, H, W)."""

    def __init__(self, obs_shape, out_dim=512):
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError("NatureCNN expects image observations")
        self.channels_last = obs_shape[-1] in (1, 3, 4) and obs_shape[0] not in (1, 3, 4)
        c = obs_shape[-1] if self.channels_last else obs_shape[0]
        h, w = (obs_shape[0], obs_shape[1]) if self.channels_last else (obs_shape[1], obs_shape[2])
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x):
        x = x.float() / 255.0
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        return self.fc(self.conv(x))


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128)):
        super().__init__()
        layers, d = [], int(in_dim)
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x.float().reshape(x.shape[0], -1))


def make_encoder(obs_shape, kind, out_dim):
    """kind: 'cnn' (NatureCNN -> out_dim) or 'mlp' (one hidden layer of out_dim, ReLU)."""
    if kind == "cnn":
        return NatureCNN(obs_shape, out_dim)
    if kind == "mlp":
        return MLPEncoder(int(np.prod(obs_shape)), (out_dim,))
    raise ValueError(f"unknown encoder kind {kind!r}")
