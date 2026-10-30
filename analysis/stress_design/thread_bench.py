"""Stress-design probe: training throughput at 1 vs 4 torch/OMP threads on this machine.

Read-only. Diagnostic training seed 401, training-env reset seed 6,100,401 (diagnostic
block). Uses each cell's selected tuned hyperparameters verbatim; nothing is saved
except out/thread_bench_<cell>_t<threads>.json. Run one process per thread setting
(OMP_NUM_THREADS must be set before torch is imported):

    OMP_NUM_THREADS=4 python -B analysis/stress_design/thread_bench.py --cell delivery/drqn --threads 4 --steps 30000
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from agents.common import baseline_plan  # noqa: E402
from agents.common.runner import build_env  # noqa: E402
from agents.common.seeding import RunRNG  # noqa: E402

TRAIN_SEED, ENV_SEED = 401, 6_100_401
OUT = os.path.join(ROOT, "analysis", "stress_design", "out")


def selected_cfg(cell):
    sel = json.load(open(os.path.join(ROOT, "run_scripts", "tuning", "results", "v3", "full",
                                      "selected_configs.json"), encoding="utf-8"))["selected"][cell]
    env, algo = cell.split("/")
    cfg = baseline_plan.resolve_config(env, algo, 1, protocol_version="v3")
    cfg["algo"]["hyperparams"] = sel["hyperparams"]
    return cfg


def bench_sb3(cfg, steps):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    cls = {"dqn": DQN, "dqn_ts": DQN, "ppo": PPO, "ppo_rm": PPO, "recurrent_ppo": RecurrentPPO}[cfg["algo"]["name"]]
    hp = dict(cfg["algo"]["hyperparams"])
    policy = hp.pop("policy")
    venv = DummyVecEnv([lambda: build_env(cfg)])
    model = cls(policy, venv, seed=RunRNG(TRAIN_SEED).torch_seed, device="cpu", verbose=0, **hp)
    venv.seed(ENV_SEED)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    return time.time() - t0


def bench_custom(cfg, steps):
    from agents.drqn.drqn_core import DRQNAgent, DRQNConfig, linear_epsilon
    from agents.qrm.qrm_core import QRMAgent, QRMConfig
    env = build_env(cfg)
    hp = cfg["algo"]["hyperparams"]
    qrm = cfg["algo"]["name"] != "drqn"
    rng = RunRNG(TRAIN_SEED)
    if qrm:
        c = QRMConfig(**hp)
        a = QRMAgent(env.observation_space, env.action_space.n, env.env.rm, c, "cpu", rng)
    else:
        c = DRQNConfig(**hp)
        a = DRQNAgent(env.observation_space, env.action_space.n, c, "cpu", rng)
    obs, info = env.reset(seed=ENV_SEED)
    head = info["rm"]["head"] if qrm else None
    a.begin_episode(obs)
    t0 = time.time()
    for step in range(1, steps + 1):
        eps = linear_epsilon(step - 1, steps, c)
        action = a.act(obs, head, eps) if qrm else a.act(obs, eps)
        nobs, r, term, trunc, info = env.step(action)
        if qrm:
            a.observe(action, r, term, trunc, nobs, info["rm"])
        else:
            a.observe(action, r, term, trunc, nobs)
        if term or trunc:
            obs, info = env.reset()
            head = info["rm"]["head"] if qrm else None
            a.begin_episode(obs)
        else:
            obs = nobs
            head = info["rm"]["next_head"] if qrm else None
        if step >= c.learning_starts and step % c.train_freq == 0:
            a.update()
        if step % c.target_update == 0:
            a.sync_target()
    return time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--steps", type=int, default=30_000)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = selected_cfg(args.cell)
    secs = bench_sb3(cfg, args.steps) if cfg["algo"]["impl"] == "sb3" else bench_custom(cfg, args.steps)
    rec = {"cell": args.cell, "threads": args.threads, "omp": os.environ.get("OMP_NUM_THREADS"),
           "torch_threads": torch.get_num_threads(), "steps": args.steps, "seconds": secs,
           "steps_per_s": args.steps / secs,
           "learning_starts": cfg["algo"]["hyperparams"].get("learning_starts")}
    print(json.dumps(rec))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"thread_bench_{args.cell.replace('/', '_')}_t{args.threads}.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
