"""Diagnostic probes after the protocol-v2 baseline (Prompt 2). Not part of any results table.

Seeds: training seed 401 (model/torch/action sampling), training-env first reset 6,100,401,
evaluation episodes 6,000,000 + i (i < 50). Outputs only under analysis/diagnostics_v3/out.
Baseline code is imported, never modified.

    python -B analysis/diagnostics_v3/run_probes.py --probe delivery_pkg1    [--steps 300000]
    python -B analysis/diagnostics_v3/run_probes.py --probe delivery_nolava  [--steps 300000]
    python -B analysis/diagnostics_v3/run_probes.py --probe maint_g999       [--steps 500000]
    python -B analysis/diagnostics_v3/run_probes.py --probe maint_baseline_eval   # baseline PPO-RM snapshots, same eval
    python -B analysis/diagnostics_v3/run_probes.py --probe colour_dense --steps 500000 --device cuda --max-minutes 40  # GPU only
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from agents.common import baseline_plan as plan  # noqa: E402
from agents.common.env_factory import make_env  # noqa: E402
from agents.common.tracking import EpisodeStats  # noqa: E402
from agents.reward_machines import RewardMachineWrapper, make_reward_machine  # noqa: E402

TRAIN_SEED = 401
TRAIN_ENV_SEED = 6_100_401
EVAL_SEEDS = [6_000_000 + i for i in range(50)]
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

PROBES = {
    "delivery_pkg1": ("delivery", "dqn_ts", {"num_packages": 1}, {}),
    "delivery_nolava": ("delivery", "dqn_ts", {"percent_death_squares": 0.0}, {}),
    "maint_g999": ("maintenance", "ppo_rm", {}, {"gamma": 0.999}),
    # optional GPU probe (not run on the laptop): python -B ... --probe colour_dense --device cuda
    "colour_dense": ("colour", "dqn_ts", {"reward_density": "dense"}, {}),
}


def build(env_key, algo, overrides):
    base = make_env(env_key, overrides=overrides or None)
    rm = make_reward_machine(env_key, base)
    if algo == "dqn_ts":
        env = RewardMachineWrapper(base, rm, obs_mode="dict" if env_key == "colour" else "append", reward_mode="env")
    else:                                            # ppo_rm
        env = RewardMachineWrapper(base, rm, obs_mode="append", reward_mode="rm")
    return EpisodeStats(env, env_key)


EV_KEYS = ("stage_exit", "final_exit", "stage_exit_incomplete", "code_collected", "service_attempt",
           "pickup", "delivery", "goal_contact", "lava")


class Log(BaseCallback):
    def __init__(self, max_seconds, window):
        super().__init__()
        self.t0, self.max_s, self.window = time.time(), max_seconds, window
        self.eps, self.events = [], Counter()
        self.win_events = {}

    def _on_step(self):
        w = (self.num_timesteps - 1) // self.window
        for inf in self.locals["infos"]:
            for name, _ in inf.get("events", []):
                if name in EV_KEYS:
                    self.events[name] += 1
                    self.win_events.setdefault(w, Counter())[name] += 1
            if "episode_record" in inf:
                self.eps.append(dict(inf["episode_record"], global_step=int(self.num_timesteps)))
        return time.time() - self.t0 < self.max_s


def evaluate(model, env, recurrent=False):
    rows = []
    for s in EVAL_SEEDS:
        obs, info = env.reset(seed=s)
        ev = Counter()
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            for name, _ in info.get("events", []):
                ev[name] += 1
            if term or trunc:
                rec = info["episode_record"]
                break
        rows.append({"seed": s, "return": rec["return_raw"], "success": rec["success"], "progress": rec["progress"],
                     "cause": rec["cause"], "length": rec["length"], **{k: ev.get(k, 0) for k in EV_KEYS}})
    return summarise_eval(rows)


def summarise_eval(rows):
    n = len(rows)
    out = {"episodes": n, "success": sum(r["success"] for r in rows) / n,
           "return_mean": float(np.mean([r["return"] for r in rows])),
           "progress_mean": float(np.mean([r["progress"] for r in rows])),
           "length_mean": float(np.mean([r["length"] for r in rows])),
           "causes": dict(Counter(r["cause"] for r in rows))}
    for k in EV_KEYS:
        out[f"{k}_per_ep"] = float(np.mean([r[k] for r in rows]))
    return out


def train_summary(cb, steps_done, window):
    eps = cb.eps
    wins = {}
    for e in eps:
        w = (e["global_step"] - 1) // window
        wins.setdefault(w, []).append(e)
    table = []
    for w in sorted(set(wins) | set(cb.win_events)):
        es = wins.get(w, [])
        ev = cb.win_events.get(w, Counter())
        table.append({"window_start": w * window, "episodes": len(es),
                      "success_rate": float(np.mean([e["success"] for e in es])) if es else None,
                      "successes": int(sum(e["success"] for e in es)),
                      "progress_mean": float(np.mean([e["progress"] for e in es])) if es else None,
                      "return_mean": float(np.mean([e["return_raw"] for e in es])) if es else None,
                      "causes": dict(Counter(e["cause"] for e in es)), **{k: ev.get(k, 0) for k in EV_KEYS}})
    return {"steps_done": int(steps_done), "episodes": len(eps), "successes": int(sum(e["success"] for e in eps)),
            "events_total": dict(cb.events), "windows": table}


def run_probe(name, steps, max_minutes, device="cpu"):
    env_key, algo, env_over, hp_over = PROBES[name]
    torch.set_num_threads(2)
    hp = plan.hyperparams(algo, env_key)
    hp.update(hp_over)
    policy = hp.pop("policy")
    venv = DummyVecEnv([lambda: build(env_key, algo, env_over)])
    cls = DQN if algo == "dqn_ts" else PPO
    if cls is PPO:
        steps = (steps // hp["n_steps"]) * hp["n_steps"]
    model = cls(policy, venv, seed=TRAIN_SEED, device=device, verbose=0, **hp)
    venv.seed(TRAIN_ENV_SEED)
    window = 100_000 if env_key == "maintenance" else 50_000
    cb = Log(max_minutes * 60, window)
    t0 = time.time()
    model.learn(steps, callback=cb, progress_bar=False)
    train_s = time.time() - t0
    model.save(os.path.join(OUT, f"{name}_policy.zip"))
    res = {"probe": name, "env": env_key, "algo": algo, "env_overrides": env_over, "hyperparam_overrides": hp_over,
           "hyperparams": {k: v for k, v in hp.items() if k != "policy_kwargs"}, "planned_steps": steps,
           "train_seconds": train_s, "stopped_early": model.num_timesteps < steps,
           "n_updates": int(model._n_updates), "train": train_summary(cb, model.num_timesteps, window)}
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)          # partial result first
    t1 = time.time()
    res["eval_greedy_50"] = evaluate(model, build(env_key, algo, env_over))
    res["eval_seconds"] = time.time() - t1
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("probe", "train_seconds", "stopped_early")}), json.dumps(res["eval_greedy_50"]),
          flush=True)


def maint_baseline_eval(step):
    """Baseline PPO-RM (gamma 0.99) snapshots at `step`, same 50 diagnostic eval seeds (10 per seed)."""
    import glob
    torch.set_num_threads(2)
    env = build("maintenance", "ppo_rm", {})
    rows = []
    for s in range(1, 6):
        d = glob.glob(os.path.join(REPO, "run_scripts", "baseline", "results", "v2", "full", "maintenance", "ppo_rm",
                                   f"seed{s}__*"))[0]
        m = PPO.load(os.path.join(d, "checkpoints", f"policy_step{step:09d}.zip"), device="cpu")
        for es in EVAL_SEEDS[(s - 1) * 10:s * 10]:
            obs, info = env.reset(seed=es)
            ev = Counter()
            while True:
                a, _ = m.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                for n_, _ in info.get("events", []):
                    ev[n_] += 1
                if term or trunc:
                    rec = info["episode_record"]
                    break
            rows.append({"seed": es, "return": rec["return_raw"], "success": rec["success"], "progress": rec["progress"],
                         "cause": rec["cause"], "length": rec["length"], **{k: ev.get(k, 0) for k in EV_KEYS}})
    out = {"snapshot_step": step, "eval_greedy_50": summarise_eval(rows)}
    json.dump(out, open(os.path.join(OUT, f"maint_baseline_eval_{step}.json"), "w"), indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True, choices=list(PROBES) + ["maint_baseline_eval"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--max-minutes", type=float, default=15)
    p.add_argument("--snapshot-step", type=int, default=409_600)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.probe == "maint_baseline_eval":
        maint_baseline_eval(a.snapshot_step)
    else:
        run_probe(a.probe, a.steps or (500_000 if a.probe.startswith("maint") else 300_000), a.max_minutes, a.device)
