"""Frozen-policy evaluation (protocol v2 section 6.4).

Evaluation uses its own environment instance, explicit episode seeds from a fixed block
(identical for every algorithm), and greedy action selection, so it consumes no random
numbers. It never writes to replay buffers, optimisers or normalisers, and keeps its own
recurrent state, reset at every episode start.
"""
import numpy as np
import torch


class SB3Policy:
    def __init__(self, model, recurrent=False):
        self.model, self.recurrent = model, recurrent

    def reset(self, obs, info):
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def act(self, obs):
        if self.recurrent:
            action, self.state = self.model.predict(obs, state=self.state, episode_start=self.episode_start,
                                                    deterministic=True)
            self.episode_start[:] = False
        else:
            action, _ = self.model.predict(obs, deterministic=True)
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


@torch.no_grad()
def evaluate(env, policy, seeds):
    """One episode per seed; returns per-episode rows (raw env return, success, cause...)."""
    rows = []
    for seed in seeds:
        obs, info = env.reset(seed=int(seed))
        policy.reset(obs, info)
        while True:
            obs, _, term, trunc, info = env.step(policy.act(obs))
            if term or trunc:
                rec = info["episode_record"]
                rows.append({"episode_seed": int(seed), "return_raw": rec["return_raw"], "length": rec["length"],
                             "success": rec["success"], "cause": rec["cause"], "progress": rec["progress"]})
                break
            policy.post_step(info)
    return rows
