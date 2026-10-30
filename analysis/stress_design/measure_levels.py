"""Stress-design probe: solvability, oracle lengths and reference returns per candidate level.

Read-only. Uses only diagnostic episode seeds (6,200,000 + i). No training, no plan or
result files are touched. Output: analysis/stress_design/out/levels_<env>_<tag>.json

    python -B analysis/stress_design/measure_levels.py --env colour --n 200
    python -B analysis/stress_design/measure_levels.py --env maintenance --n 200 --levels "[[1,1],[1,2]]"

Policies per level:
  oracle   scripted privileged controller (temporal_comp_gym/oracles.py)
  random   uniform random actions, generator seeded by the episode seed
  null     a fixed "do nothing" action where the task has one (maintenance: zero thrust and
           torque); delivery and colour have no no-op action, so it is skipped there
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from agents.common.env_factory import env_kwargs, make_env_from_kwargs  # noqa: E402
from temporal_comp_gym.oracles import ORACLES  # noqa: E402

SEED_BASE = 6_200_000          # diagnostic episode-seed block (protocol: 6,000,000+)
OUT = os.path.join(ROOT, "analysis", "stress_design", "out")

AXES = {
    "delivery": ("num_packages", [1, 2, 3, 4, 5, 6, 7]),
    "colour": ("retention_delay_steps", [0, 2, 8, 16, 24, 32, 48, 64]),
    "maintenance": ("readiness_delay_range", [[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]]),
}


def required_count(env_key, core):
    if env_key == "delivery":
        return int(core.num_packages)
    if env_key == "colour":
        return len(core.sequence)
    return len(core.issued_codes)


def rollout(env_key, env, policy, seed):
    core = env.unwrapped
    obs, info = env.reset(seed=seed)
    req = required_count(env_key, core)
    rng = np.random.default_rng(seed)
    oracle = None
    if policy == "oracle":
        oracle = ORACLES[env_key](env)
        oracle.reset()
    total, steps, progress = 0.0, 0, 0
    ev_steps = {}                       # first step index of selected events
    held_since = {}                     # maintenance: code -> step collected
    hold_spans = []                     # maintenance: steps from collection to service
    ready = False
    first_delivery = None
    cue_steps = [0] if env_key == "colour" else []
    while True:
        if policy == "oracle":
            a = oracle.act()
        elif policy == "random":
            if env_key == "maintenance":
                a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            else:
                a = int(rng.integers(4))
        else:
            a = np.zeros(2, dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        for name, arg in info["events"]:
            ev_steps.setdefault(name, steps)
            if env_key == "delivery" and name == "delivery":
                progress += 1
                if first_delivery is None:
                    first_delivery = steps
            elif env_key == "colour":
                if name == "room_reset":
                    progress += 1
                elif name == "cue":
                    cue_steps.append(steps)
            elif env_key == "maintenance":
                if name == "code_collected":
                    held_since[arg] = steps
                elif name == "readiness_on":
                    ready = True
                elif name == "service_attempt" and ready and arg in held_since:
                    hold_spans.append(steps - held_since.pop(arg))
                    progress += 1
        if term or trunc:
            break
    success = bool(info.get("is_success", False))
    if env_key == "colour" and success:
        progress += 1
    rec = {"seed": seed, "return": total, "length": steps, "success": success,
           "cause": info.get("termination_cause"), "progress": progress, "required": req}
    if env_key == "delivery":
        rec["layout_attempts"] = int(getattr(core, "layout_attempts", 0))
        # memory span: steps from the first delivery to the episode end (the count must be held)
        rec["span_first_delivery_to_end"] = None if first_delivery is None else steps - first_delivery
    elif env_key == "colour":
        nav = ev_steps.get("nav_start")
        rec["nav_start_step"] = nav
        rec["last_cue_step"] = cue_steps[-1]
        rec["span_first_cue_to_end"] = steps if success else None
    else:
        rec["hold_spans"] = hold_spans
        rec["readiness_target"] = int(core.readiness_target)
    return rec


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else None


def summarise(rows, env_key):
    lens = [r["length"] for r in rows]
    s = {
        "n": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "return_mean": float(np.mean([r["return"] for r in rows])),
        "return_sd": float(np.std([r["return"] for r in rows])),
        "length_mean": float(np.mean(lens)), "length_p50": pct(lens, 50), "length_p95": pct(lens, 95),
        "length_max": int(max(lens)),
        "progress_mean": float(np.mean([r["progress"] for r in rows])),
        "required_mean": float(np.mean([r["required"] for r in rows])),
        "progress_fraction_mean": float(np.mean([r["progress"] / r["required"] for r in rows])),
        "causes": {},
    }
    for r in rows:
        s["causes"][r["cause"]] = s["causes"].get(r["cause"], 0) + 1
    if env_key == "delivery":
        att = [r["layout_attempts"] for r in rows]
        s["layout_attempts_mean"] = float(np.mean(att))
        s["layout_resampled_fraction"] = float(np.mean([a > 1 for a in att]))
        sp = [r["span_first_delivery_to_end"] for r in rows if r["span_first_delivery_to_end"] is not None]
        s["span_first_delivery_to_end_mean"] = float(np.mean(sp)) if sp else None
    elif env_key == "colour":
        s["nav_start_mean"] = float(np.mean([r["nav_start_step"] for r in rows if r["nav_start_step"]]))
        s["required_values"] = sorted({r["required"] for r in rows})
    else:
        spans = [x for r in rows for x in r["hold_spans"]]
        s["hold_span_mean"] = float(np.mean(spans)) if spans else None
        s["hold_span_p95"] = pct(spans, 95)
        s["readiness_target_counts"] = {str(k): sum(r["readiness_target"] == k for r in rows) for k in (1, 2, 3)}
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=sorted(AXES))
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--levels", default=None, help="JSON list of axis values (default: built-in candidates)")
    p.add_argument("--policies", default="oracle,random,null")
    p.add_argument("--tag", default="all")
    p.add_argument("--axis", default=None, help="env kwarg to vary instead of the primary axis")
    p.add_argument("--extra", default=None, help="JSON dict of further env overrides (e.g. max_steps)")
    args = p.parse_args()
    key, default_levels = AXES[args.env]
    if args.axis:
        key = args.axis
    levels = json.loads(args.levels) if args.levels else default_levels
    policies = [x for x in args.policies.split(",") if x]
    if args.env != "maintenance":
        policies = [x for x in policies if x != "null"]
    seeds = [SEED_BASE + i for i in range(args.n)]
    out = {"env": args.env, "axis": key, "seed_block": f"{SEED_BASE}+i, i<{args.n}", "levels": []}
    for value in levels:
        kw = env_kwargs(args.env, overrides={key: value, **(json.loads(args.extra) if args.extra else {})})
        entry = {"value": value, "env_kwargs": kw, "max_steps": kw["max_steps"]}
        try:
            env = make_env_from_kwargs(args.env, kw, debug_info=False)
        except Exception as exc:      # invalid level: record, do not crash the sweep
            entry["error"] = repr(exc)
            out["levels"].append(entry)
            continue
        for pol in policies:
            t0 = time.time()
            rows, errors = [], []
            for s in seeds:
                try:
                    rows.append(rollout(args.env, env, pol, s))
                except RuntimeError as exc:   # e.g. no solvable delivery layout
                    errors.append({"seed": s, "error": repr(exc)})
            entry[pol] = summarise(rows, args.env) if rows else None
            entry[pol + "_errors"] = errors
            entry[pol + "_seconds"] = round(time.time() - t0, 1)
            print(args.env, value, pol, json.dumps({k: entry[pol][k] for k in
                  ("success_rate", "length_p95", "return_mean", "progress_fraction_mean")} if rows else None),
                  f"errors={len(errors)}", flush=True)
        env.close()
        out["levels"].append(entry)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"levels_{args.env}_{args.tag}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
