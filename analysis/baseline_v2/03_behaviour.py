"""Step 3: behaviour of the saved policy snapshots (read-only; no training).

For every cell and seed, loads policy snapshots (10%, 50%, 100% of the budget) and rolls out
  greedy       epsilon = 0 / deterministic=True (the evaluation policy)
  exploratory  epsilon = 0.05 (DQN family, the training floor) / stochastic actions (PPO family)
on the first N periodic-evaluation episode seeds (1,000,000+i). Exploration noise comes from
the diagnostic stream 6,000,000+ (see REPORT.md), never from a reserved range.

Per episode it records return, discounted return, length, cause, progress, action
distribution, event counts, Q-values (DQN family) and task-specific behaviour features.

    python -B analysis/baseline_v2/03_behaviour.py --env delivery [--episodes 10] [--fracs 0.1 0.5 1.0]
"""
import argparse
import dataclasses
import hashlib
import json
import os
import time
from collections import Counter

import numpy as np
import torch

from common import CELLS, DIAG_EPISODE_SEED_BASE, OUT, SEEDS, read_json, run_dirs, write_json
from agents.common.runner import build_env
from agents.common.seeding import EVAL_EPISODE_SEED_BASE

GAMMA = 0.99
torch.set_num_threads(2)


# ---------------------------------------------------------------------- policies
class SB3DQN:
    def __init__(self, path, cfg):
        from stable_baselines3 import DQN
        self.m = DQN.load(path, device="cpu")

    def reset(self, obs, info):
        pass

    def q(self, obs):
        with torch.no_grad():
            t, _ = self.m.policy.obs_to_tensor(obs)
            return self.m.q_net(t)[0].numpy()

    def post_step(self, info):
        pass


class SB3PPO:
    def __init__(self, path, cfg, recurrent):
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import PPO
        self.m = (RecurrentPPO if recurrent else PPO).load(path, device="cpu")
        self.recurrent = recurrent

    def reset(self, obs, info):
        self.state, self.start = None, np.ones((1,), dtype=bool)

    def act(self, obs, stochastic):
        if self.recurrent:
            a, self.state = self.m.predict(obs, state=self.state, episode_start=self.start, deterministic=not stochastic)
            self.start[:] = False
        else:
            a, _ = self.m.predict(obs, deterministic=not stochastic)
        return a

    def post_step(self, info):
        pass


class DRQN:
    def __init__(self, path, cfg, env):
        from agents.drqn.drqn_core import DRQNAgent, DRQNConfig
        from agents.common.seeding import RunRNG
        c = dataclasses.replace(DRQNConfig(**cfg["algo"]["hyperparams"]), replay_frames=16)
        self.a = DRQNAgent(env.observation_space, env.action_space.n, c, "cpu", RunRNG(401))
        self.a.q.load_state_dict(torch.load(path, map_location="cpu"))

    def reset(self, obs, info):
        self.h = self.a.zero_hidden()
        self._q = None

    def q(self, obs):                       # advances the hidden state (called exactly once per step)
        q, self.h = self.a.policy_step(obs, self.h)
        return q

    def post_step(self, info):
        pass


class QRM:
    def __init__(self, path, cfg, env):
        from agents.qrm.qrm_core import QRMAgent, QRMConfig
        from agents.common.seeding import RunRNG
        c = dataclasses.replace(QRMConfig(**cfg["algo"]["hyperparams"]), replay_frames=16)
        self.a = QRMAgent(env.observation_space, env.action_space.n, env.env.rm, c, "cpu", RunRNG(401))
        self.a.q.load_state_dict(torch.load(path, map_location="cpu"))

    def reset(self, obs, info):
        self.head = info["rm"]["head"]

    def q(self, obs):
        return self.a.q_values(obs, self.head)

    def post_step(self, info):
        self.head = info["rm"]["next_head"]


def load_policy(algo, path, cfg, env):
    if algo in ("dqn", "dqn_ts"):
        return SB3DQN(path, cfg)
    if algo in ("ppo", "ppo_rm", "recurrent_ppo"):
        return SB3PPO(path, cfg, algo == "recurrent_ppo")
    if algo == "drqn":
        return DRQN(path, cfg, env)
    return QRM(path, cfg, env)


# ---------------------------------------------------------------------- task features
def core(env):
    return env.unwrapped


def feat_init(env_key, env):
    c = core(env)
    if env_key == "delivery":
        return {"cells": Counter(), "states": {}, "first_repeat": None, "pickups": 0}
    if env_key == "colour":
        return {"positions": Counter(), "max_disp": 0.0, "nav_steps": 0, "wall_touches": 0,
                "first_nav_q": None}
    return {"boundary_steps": 0, "offpath_steps": 0, "stages": 0, "codes": 0, "services": 0,
            "speed": [], "thrust": [], "torque": [], "y": [], "x": [], "brake": 0, "angle_abs": []}


def feat_step(env_key, env, f, t, action, info, qvec):
    c = core(env)
    ev = [e[0] for e in info.get("events", [])]
    if env_key == "delivery":
        pos = tuple(int(v) for v in c.agent_pos)
        f["cells"][pos] += 1
        f["pickups"] += ev.count("pickup")
        grid_sig = hashlib.md5(c.grid.encode().tobytes()).hexdigest()[:8]
        s = (pos, c.health, c.carried_packages, c.delivered_packages, grid_sig)
        if f["first_repeat"] is None:
            if s in f["states"]:
                f["first_repeat"] = (t, t - f["states"][s])   # (step, cycle length)
            else:
                f["states"][s] = t
    elif env_key == "colour":
        if c.current_phase == "navigation":
            f["nav_steps"] += 1
            p = (round(c.agent.x), round(c.agent.y))
            f["positions"][p] += 1
            f["max_disp"] = max(f["max_disp"], abs(c.agent.x - c.agent.start_x), abs(c.agent.y - c.agent.start_y))
            if f["first_nav_q"] is None and qvec is not None:
                f["first_nav_q"] = float(np.max(qvec))
        f["wall_touches"] += ev.count("wall_touch")
    else:
        b = c.rocket.body
        f["boundary_steps"] += ev.count("boundary_contact")
        f["offpath_steps"] += ev.count("off_path")
        f["stages"] += ev.count("stage_exit")
        f["codes"] += ev.count("code_collected")
        f["services"] += ev.count("service_attempt")
        f["speed"].append(float(np.hypot(b.velocity.x, b.velocity.y)) / c.cell_size)
        a = np.asarray(action, dtype=float).reshape(-1)
        f["thrust"].append(float(np.clip(a[0], -1, 1)))
        f["torque"].append(float(np.clip(a[1], -1, 1)))
        f["brake"] += int(a[0] < -0.5)
        f["y"].append(float(b.position.y) / c.cell_size)
        f["x"].append(float(b.position.x) / c.cell_size)
        f["angle_abs"].append(abs(float(b.angle)))


def feat_final(env_key, f, length):
    if env_key == "delivery":
        top = f["cells"].most_common(1)[0][1] if f["cells"] else 0
        return {"unique_cells": len(f["cells"]), "top_cell_frac": top / max(1, length), "pickups": f["pickups"],
                "first_state_repeat_step": None if f["first_repeat"] is None else f["first_repeat"][0],
                "cycle_len": None if f["first_repeat"] is None else f["first_repeat"][1]}
    if env_key == "colour":
        top = f["positions"].most_common(1)[0][1] if f["positions"] else 0
        return {"nav_steps": f["nav_steps"], "unique_nav_positions": len(f["positions"]),
                "top_position_frac": top / max(1, f["nav_steps"]), "max_disp_blocks": f["max_disp"] / (800 / 15),
                "wall_touches": f["wall_touches"], "first_nav_maxq": f["first_nav_q"]}
    n = max(1, length)
    return {"boundary_frac": f["boundary_steps"] / n, "offpath_frac": f["offpath_steps"] / n, "stages": f["stages"],
            "codes": f["codes"], "service_attempts": f["services"], "mean_speed_cells_s": float(np.mean(f["speed"])),
            "frac_slow": float(np.mean(np.array(f["speed"]) < 0.5)), "thrust_mean": float(np.mean(f["thrust"])),
            "thrust_pos_frac": float(np.mean(np.array(f["thrust"]) > 0)), "brake_frac": f["brake"] / n,
            "torque_mean": float(np.mean(f["torque"])), "angle_abs_mean": float(np.mean(f["angle_abs"])),
            "y_range": float(np.ptp(f["y"])), "x_range": float(np.ptp(f["x"])), "y_last": f["y"][-1], "x_last": f["x"][-1]}


# ---------------------------------------------------------------------- rollouts
def rollout(env_key, algo, env, pol, ep_seed, mode, rng, eps=0.05, record_trace=False):
    obs, info = env.reset(seed=int(ep_seed))
    pol.reset(obs, info)
    f = feat_init(env_key, env)
    acts, events = [], Counter()
    q0, qmax, disc, t = None, -np.inf, 0.0, 0
    trace = []
    while True:
        qvec = None
        if hasattr(pol, "q"):
            qvec = pol.q(obs)
            if q0 is None:
                q0 = float(np.max(qvec))
            qmax = max(qmax, float(np.max(qvec)))
            a = int(np.argmax(qvec))
            if mode == "explore" and rng.random() < eps:
                a = int(rng.integers(env.action_space.n))
        else:
            a = pol.act(obs, stochastic=(mode == "explore"))
        obs, r, term, trunc, info = env.step(a)
        disc += (GAMMA ** t) * float(info.get("raw_reward", r))
        t += 1
        acts.append(a if np.ndim(a) == 0 else tuple(np.round(np.asarray(a, float).reshape(-1), 2)))
        for e in info.get("events", []):
            events[e[0]] += 1
        feat_step(env_key, env, f, t, a, info, qvec)
        if record_trace and len(trace) < 400:
            trace.append({"t": t, "a": a if np.ndim(a) == 0 else [float(v) for v in np.asarray(a).reshape(-1)],
                          "r": float(r), "events": info.get("events", [])})
        if term or trunc:
            rec = info["episode_record"]
            break
        pol.post_step(info)
    out = {"episode_seed": int(ep_seed), "mode": mode, "return": rec["return_raw"], "disc_return": disc,
           "length": rec["length"], "cause": rec["cause"], "success": rec["success"], "progress": rec["progress"],
           "events": dict(events), "q0_max": q0, "q_max_episode": None if qmax == -np.inf else qmax}
    if env_key != "maintenance":
        cnt = Counter(acts)
        out["action_frac"] = {int(k): v / len(acts) for k, v in sorted(cnt.items())}
        out["dominant_action_frac"] = max(cnt.values()) / len(acts)
    out.update(feat_final(env_key, f, rec["length"]))
    if record_trace:
        out["trace"] = trace
    return out


def snapshot_path(run_dir, algo, step):
    ext = "zip" if algo in ("dqn", "dqn_ts", "ppo", "ppo_rm", "recurrent_ppo") else "pt"
    return os.path.join(run_dir, "checkpoints", f"policy_step{step:09d}.{ext}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(CELLS))
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = p.parse_args()
    runs = run_dirs()
    env_key = args.env
    algos = args.algos or CELLS[env_key]
    episode_seeds = [EVAL_EPISODE_SEED_BASE["periodic"] + i for i in range(args.episodes)]
    results = []
    t0 = time.time()
    for algo in algos:
        for seed in args.seeds:
            d = runs[(env_key, algo, seed)]
            cfg = read_json(os.path.join(d, "config.resolved.json"))
            budget = cfg["budget"]["env_steps"]
            env = build_env(cfg)
            for frac in args.fracs:
                step = int(round(frac * budget))
                path = snapshot_path(d, algo, step)
                if not os.path.exists(path):
                    results.append({"algo": algo, "seed": seed, "snapshot_step": step, "missing": True})
                    continue
                pol = load_policy(algo, path, cfg, env)
                for mode in ("greedy", "explore"):
                    # diagnostic stream (6,000,000+), distinct per algo/seed/snapshot
                    rng = np.random.default_rng([DIAG_EPISODE_SEED_BASE, seed, step, 0 if mode == "greedy" else 1])
                    torch.manual_seed(DIAG_EPISODE_SEED_BASE + seed)
                    for i, es in enumerate(episode_seeds):
                        r = rollout(env_key, algo, env, pol, es, mode, rng,
                                    record_trace=(frac == 1.0 and i == 0 and seed == 1))
                        r.update(algo=algo, seed=seed, snapshot_step=step, frac=frac)
                        results.append(r)
                print(f"{env_key}/{algo} seed{seed} step{step}: done ({time.time() - t0:.0f}s)", flush=True)
            env.close()
            write_json(os.path.join(OUT, f"behaviour_{env_key}.json"), {"env": env_key, "episode_seeds": episode_seeds,
                                                                         "epsilon_explore": 0.05, "episodes": results})
    write_json(os.path.join(OUT, f"behaviour_{env_key}.json"), {"env": env_key, "episode_seeds": episode_seeds,
                                                                 "epsilon_explore": 0.05, "episodes": results})
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
