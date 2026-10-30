"""Frozen-policy evaluation (protocol v2 section 6.4; v3 decision D5).

Evaluation uses its own environment instance, explicit episode seeds from a fixed block
(identical for every algorithm), and never writes to replay buffers, optimisers or
normalisers. It keeps its own recurrent state, reset at every episode start.

Headline evaluation (v2 and v3): greedy actions, no random numbers at all.
Secondary "noisy" evaluation (v3 only): training-time action noise. Value-based agents take a
uniformly random action with probability epsilon (their Q-values are still computed every
step, so recurrent state advances exactly as during training); PPO-family agents sample from
their policy distribution. All randomness comes from a per-episode generator seeded by the
episode seed, and torch's global RNG is forked and restored around each episode, so the
secondary evaluation is reproducible, paired across algorithms, and leaves every training
random stream untouched.
"""
import numpy as np
import torch


class SB3Policy:
    def __init__(self, model, recurrent=False, deterministic=True):
        self.model, self.recurrent, self.deterministic = model, recurrent, deterministic

    def reset(self, obs, info):
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def act(self, obs):
        if self.recurrent:
            action, self.state = self.model.predict(obs, state=self.state, episode_start=self.episode_start,
                                                    deterministic=self.deterministic)
            self.episode_start[:] = False
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action) if np.ndim(action) == 0 else action

    def post_step(self, info):
        pass


class DRQNPolicy:
    def __init__(self, agent):
        self.agent = agent

    def reset(self, obs, info):
        self.hidden = self.agent.zero_hidden()

    def act(self, obs):
        q, self.hidden = self.agent.policy_step(obs, self.hidden)
        return int(np.argmax(q))

    def post_step(self, info):
        pass


class QRMPolicy:
    """Greedy w.r.t. the Q head of the RM state tracked in the eval env's own RM wrapper."""

    def __init__(self, agent):
        self.agent = agent

    def reset(self, obs, info):
        self.head = info["rm"]["head"]

    def act(self, obs):
        return int(np.argmax(self.agent.q_values(obs, self.head)))

    def post_step(self, info):
        self.head = info["rm"]["next_head"]


class EpsilonPolicy:
    """Wraps a greedy value-based policy: the greedy action is always computed first (so any
    recurrent state advances), then replaced by a uniform random action with probability eps."""

    def __init__(self, greedy, n_actions, epsilon):
        self.greedy, self.n_actions, self.epsilon = greedy, int(n_actions), float(epsilon)
        self.rng = None

    def set_episode_rng(self, rng):
        self.rng = rng

    def reset(self, obs, info):
        self.greedy.reset(obs, info)

    def act(self, obs):
        action = self.greedy.act(obs)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return action

    def post_step(self, info):
        self.greedy.post_step(info)


def _record_row(seed, rec):
    row = {"episode_seed": int(seed), "return_raw": rec["return_raw"], "length": rec["length"],
           "success": rec["success"], "cause": rec["cause"], "progress": rec["progress"],
           "stage_exits": rec.get("stage_exits"), "wall_touches": rec.get("wall_touches")}
    if "required" in rec:                              # stress phase only (STRESS_PROTOCOL.md 9.1)
        from agents.common.run_io import STRESS_EPISODE_FIELDS
        row.update({k: rec[k] for k in STRESS_EPISODE_FIELDS})
    return row


def _run_episode(env, policy, seed):
    obs, info = env.reset(seed=int(seed))
    policy.reset(obs, info)
    while True:
        obs, _, term, trunc, info = env.step(policy.act(obs))
        if term or trunc:
            return _record_row(seed, info["episode_record"])
        policy.post_step(info)


@torch.no_grad()
def evaluate(env, policy, seeds, rng_offset=None):
    """One episode per seed; returns per-episode rows (raw env return, success, cause...).

    rng_offset=None: deterministic policy, no random numbers drawn.
    rng_offset=int: the policy may sample; each episode gets np.random.default_rng([seed,
    rng_offset]) (passed to policies that accept it) and a torch RNG forked from the global
    state and seeded from that generator, restored afterwards.
    """
    rows = []
    for seed in seeds:
        if rng_offset is None:
            rows.append(_run_episode(env, policy, seed))
            continue
        rng = np.random.default_rng([int(seed), int(rng_offset)])
        if hasattr(policy, "set_episode_rng"):
            policy.set_episode_rng(rng)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(rng.integers(2 ** 31 - 1)))
            rows.append(_run_episode(env, policy, seed))
    return rows
