"""Task solvability report for protocol v2 (no learning involved).

Runs the scripted privileged oracle and a uniform-random policy on the baseline
final-evaluation episode seeds of each environment, checks every step's reward-machine
reward against the env reward, and writes a JSON report. The oracle and random numbers are
the normalisation references in the protocol (section 5).

    python run_scripts/baseline/validate_tasks.py [--episodes 200] [--random-episodes 200]
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from agents.common.env_factory import PROTOCOL_VERSION, env_kwargs, make_env  # noqa: E402
from agents.common.seeding import eval_episode_seeds  # noqa: E402
from agents.reward_machines import RewardMachineWrapper, make_reward_machine  # noqa: E402
from temporal_comp_gym.oracles import ORACLES  # noqa: E402


def run(key, policy, seeds):
    base = make_env(key, debug_info=(policy == "oracle"))
    env = RewardMachineWrapper(base, make_reward_machine(key, base))   # raises on any reward mismatch
    oracle = ORACLES[key](env)
    rng = np.random.default_rng(0)
    env.action_space.seed(0)
    rows = []
    for seed in seeds:
        env.reset(seed=seed)
        oracle.reset()
        ret, length = 0.0, 0
        while True:
            a = oracle.act() if policy == "oracle" else env.action_space.sample()
            _, r, term, trunc, info = env.step(a)
            ret += r
            length += 1
            if term or trunc:
                break
        rows.append({"seed": seed, "return": ret, "length": length,
                     "success": bool(info["is_success"]), "cause": info["termination_cause"]})
    lengths = np.array([r["length"] for r in rows])
    causes = {}
    for r in rows:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "return_mean": float(np.mean([r["return"] for r in rows])),
        "return_std": float(np.std([r["return"] for r in rows])),
        "length_p50": float(np.percentile(lengths, 50)),
        "length_p95": float(np.percentile(lengths, 95)),
        "length_max": int(lengths.max()),
        "causes": causes,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--random-episodes", type=int, default=200)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "run_scripts", "baseline", "validation",
                                                 f"task_validation_{PROTOCOL_VERSION}.json"))
    args = p.parse_args()
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        rev, dirty = None, None
    report = {"protocol_version": PROTOCOL_VERSION, "git_rev": rev, "git_dirty": dirty,
              "python": sys.version.split()[0], "platform": platform.platform(),
              "eval_seed_block": "final", "environments": {}}
    for key in ("delivery", "colour", "maintenance"):
        t = time.time()
        res = {"env_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in env_kwargs(key).items()},
               "oracle": run(key, "oracle", eval_episode_seeds("final", args.episodes)),
               "random": run(key, "random", eval_episode_seeds("final", args.random_episodes))}
        horizon = res["env_kwargs"]["max_steps"]
        res["horizon"] = horizon
        res["oracle_p95_over_horizon"] = res["oracle"]["length_p95"] / horizon
        report["environments"][key] = res
        print(f"{key}: oracle success {res['oracle']['success_rate']:.3f} "
              f"(p95 length {res['oracle']['length_p95']:.0f} / {horizon}), "
              f"random success {res['random']['success_rate']:.3f}, {time.time() - t:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")
    ok = all(report["environments"][k]["oracle"]["success_rate"] >= (0.95 if k == "maintenance" else 1.0)
             and report["environments"][k]["oracle_p95_over_horizon"] <= 0.5 for k in report["environments"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
