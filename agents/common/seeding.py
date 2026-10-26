"""Seed sets and per-run random streams (protocol v2 section 6.3).

Training seeds, evaluation episode seeds and the later tuning/stress/test sets live in
disjoint ranges. Every random stream a run uses is derived from its training seed with a
fixed, named child of numpy's SeedSequence, so adding a stream never changes another one.
"""
import random

import numpy as np

BASELINE_TRAIN_SEEDS = (1, 2, 3, 4, 5)
BASELINE_EXPANSION_SEEDS = (6, 7, 8, 9, 10)

TRAIN_SEED_RANGES = {
    "baseline": range(1, 11),
    "tuning": range(101, 200),
    "stress": range(201, 300),
    "final_test": range(301, 400),
}

EVAL_EPISODE_SEED_BASE = {
    "periodic": 1_000_000,          # baseline periodic evaluation
    "final": 2_000_000,             # baseline final evaluation
    "tuning_validation": 3_000_000,
    "stress_eval": 4_000_000,
    "final_test": 5_000_000,
}
EVAL_BLOCK_SIZE = 1_000_000
TRAIN_ENV_SEED_BASE = 10_000_000

# Fixed child index per stream name. Never renumber existing entries.
_STREAMS = {
    "torch": 0,
    "agent_numpy": 1,
    "python": 2,
    "action_space": 3,
    "replay": 4,
    "exploration": 5,
}


def eval_episode_seeds(kind, n):
    """The first n episode seeds of an evaluation block (identical for every algorithm)."""
    base = EVAL_EPISODE_SEED_BASE[kind]
    if not 0 < n <= EVAL_BLOCK_SIZE:
        raise ValueError("n must be in (0, EVAL_BLOCK_SIZE]")
    return [base + i for i in range(n)]


def train_env_seed(train_seed):
    """Seed for the training env's first reset; later resets continue the env's own RNG."""
    return TRAIN_ENV_SEED_BASE + 1000 * int(train_seed)


def stream_seed(train_seed, name):
    """A 32-bit integer seed for a named stream of a training run."""
    ss = np.random.SeedSequence(entropy=int(train_seed), spawn_key=(_STREAMS[name],))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


class RunRNG:
    """Independent generators for one training run."""

    def __init__(self, train_seed):
        self.train_seed = int(train_seed)
        self.numpy = np.random.default_rng(stream_seed(train_seed, "agent_numpy"))
        self.replay = np.random.default_rng(stream_seed(train_seed, "replay"))
        self.exploration = np.random.default_rng(stream_seed(train_seed, "exploration"))
        self.python = random.Random(stream_seed(train_seed, "python"))
        self.torch_seed = stream_seed(train_seed, "torch")
        self.action_space_seed = stream_seed(train_seed, "action_space")


def seed_torch(seed):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
