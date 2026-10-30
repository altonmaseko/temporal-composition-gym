"""Stress-test plan v3 (decision D11, run_scripts/stress/STRESS_PROTOCOL.md).

The plan is built deterministically from the frozen protocol and the published tuning
selection. Nothing is sampled: every hyperparameter is copied verbatim from
run_scripts/tuning/results/v3/full/selected_configs.json, and plans are compared as exact
JSON values (never via file bytes, so Windows/Linux line endings cannot matter).

Run identity: (env, algo, level, training seed). Machine roles a/b/c each run whole
"sets" (all 11 cells at one level and seed) in rounds, as in STRESS_PROTOCOL.md section 10.
"""
import copy
import hashlib
import json
from pathlib import Path

from agents.common import baseline_plan as base
from agents.common.env_factory import (STRESS_AXES, STRESS_LEVEL_OVERRIDES, STRESS_LEVELS,
                                       stress_env_kwargs, stress_within_steps)
from agents.common.seeding import TRAIN_SEED_RANGES, eval_episode_seeds, train_env_seed
from agents.common.tuning_plan import CELLS, validate_hp

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "run_scripts" / "stress" / "stress_plan_v3.json"
RESULTS = ROOT / "run_scripts" / "stress" / "results"
SELECTED_PATH = ROOT / "run_scripts" / "tuning" / "results" / "v3" / "full" / "selected_configs.json"
PHASE = "stress"
SCHEMA = 1

DEFAULT_LEVEL = "default"
DEFAULT_SEEDS = (301, 302, 303, 304, 305)
STRESS_SEEDS = (201, 202, 203)
# level kind -> (training seed range, periodic eval block, final eval block)
SEED_RULES = {
    "default": ("final_test", "final_test", "final_test_terminal"),
    "stress": ("stress", "stress_eval", "stress_terminal"),
}
THREADS = 1                       # P16: passed explicitly to every run and recorded
DEVICE_POLICY = {"delivery": "cpu", "colour": "cuda", "maintenance": "cpu"}

# Planning hours per full run (STRESS_PROTOCOL.md section 10; not part of the plan or its hash).
PLANNING_HOURS = {("delivery", "dqn"): 0.18, ("delivery", "dqn_ts"): 0.19, ("delivery", "drqn"): 0.55,
                  ("delivery", "qrm"): 0.13, ("colour", "dqn"): 1.16, ("colour", "dqn_ts"): 1.23,
                  ("colour", "drqn"): 3.71, ("colour", "qrm_cue"): 1.12, ("maintenance", "ppo"): 0.54,
                  ("maintenance", "recurrent_ppo"): 1.28, ("maintenance", "ppo_rm"): 0.58}
# Extra evaluation time at the raised 5,000-step maintenance limit (+25% of evaluation hours).
RAISED_LIMIT_EXTRA_HOURS = {"ppo": 0.05, "recurrent_ppo": 0.12, "ppo_rm": 0.07}


def planning_hours(env, algo, level):
    h = PLANNING_HOURS[(env, algo)]
    if env == "maintenance" and level in ("hard1", "hard2"):
        h += RAISED_LIMIT_EXTRA_HOURS[algo]
    return h


# Order inside a set: CPU cells first, colour DRQN (longest) last (section 10).
CELL_ORDER = [("delivery", "dqn"), ("delivery", "dqn_ts"), ("delivery", "drqn"), ("delivery", "qrm"),
              ("maintenance", "ppo"), ("maintenance", "recurrent_ppo"), ("maintenance", "ppo_rm"),
              ("colour", "dqn"), ("colour", "dqn_ts"), ("colour", "qrm_cue"), ("colour", "drqn")]
ROLES = ("a", "b", "c")
# Rounds of whole sets per role (section 10).
ROLE_SETS = {
    "a": [("default", 301), ("easy", 201), ("default", 304), ("hard1", 202), ("hard1", 203)],
    "b": [("default", 302), ("hard1", 201), ("default", 305), ("hard2", 202), ("hard2", 203)],
    "c": [("default", 303), ("hard2", 201), ("easy", 202), ("easy", 203)],
}
# Balance: these colour DRQN runs move from the end of A's and B's queues to the end of C's.
MOVED_TO_C = [("hard1", 203, "colour", "drqn"), ("hard2", 203, "colour", "drqn")]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                          .encode("utf-8")).hexdigest()


def level_kind(level):
    if level not in STRESS_LEVELS:
        raise ValueError(f"unknown stress level {level!r}; expected one of {STRESS_LEVELS}")
    return "default" if level == DEFAULT_LEVEL else "stress"


def check_seed(level, seed):
    """Final-comparison seeds only at the default level, stress seeds only at the other levels."""
    seed_range = SEED_RULES[level_kind(level)][0]
    if int(seed) not in TRAIN_SEED_RANGES[seed_range]:
        raise ValueError(f"training seed {seed} is not allowed at stress level {level!r} "
                         f"(requires {TRAIN_SEED_RANGES[seed_range]})")


# --------------------------------------------------------------------------- tuned learners
def load_selected(path=SELECTED_PATH):
    """{cell: {'candidate', 'hyperparams'}} plus the tuning plan hash, verbatim from the file."""
    sel = json.loads(Path(path).read_text(encoding="utf-8"))
    if (sel.get("phase"), sel.get("mode"), sel.get("protocol_version")) != ("tuning", "full", "v3"):
        raise ValueError("selected_configs.json is not the full tuning v3 selection")
    if sel.get("complete") is not True or sel.get("usable_for_final_comparison") is not True:
        raise ValueError("tuning selection is incomplete or not marked usable")
    cells = {}
    for env, algo in CELLS:
        entry = sel["selected"][f"{env}/{algo}"]
        cells[f"{env}/{algo}"] = {"candidate": int(entry["candidate"]),
                                  "hyperparams": copy.deepcopy(entry["hyperparams"])}
    if set(sel["selected"]) != set(cells):
        raise ValueError("selected_configs.json cells differ from the 11 tuning cells")
    return {"tuning_plan_hash": sel["plan_hash"], "cells": cells}


# --------------------------------------------------------------------------- configuration
def resolve_config(env, algo, level, seed, hp, candidate, mode="full"):
    """Complete resolved configuration of one stress run (JSON-able)."""
    if (env, algo) not in CELLS:
        raise ValueError(f"{env}/{algo} is not a stress cell")
    if mode not in ("full", "smoke"):
        raise ValueError("mode must be 'full' or 'smoke'")
    check_seed(level, seed)
    validate_hp(env, algo, hp)
    seed = int(seed)
    _, periodic_block, final_block = SEED_RULES[level_kind(level)]
    cfg = base.resolve_config(env, algo, 1, protocol_version="v3")
    cfg["phase"], cfg["mode"] = PHASE, mode
    kwargs = stress_env_kwargs(env, level)
    cfg["env"]["kwargs"] = base._jsonable(kwargs)
    if level != DEFAULT_LEVEL:
        cfg["env"]["variant"] = f"stress-{level}"
    cfg["algo"]["hyperparams"] = copy.deepcopy(hp)
    cfg["seed"] = {"train_seed": seed, "train_env_reset_seed": train_env_seed(seed)}
    cfg["eval"].update(periodic_seed_block=periodic_block, final_seed_block=final_block)
    cfg["wandb"]["group"] = "stress-v3" + ("-smoke" if mode == "smoke" else "")
    axis = STRESS_AXES[env]
    cfg["stress"] = {"schema": SCHEMA, "level": level, "axis": axis, "axis_value": base._jsonable(kwargs[axis]),
                     "within_steps": stress_within_steps(env, level), "tuned_candidate": int(candidate),
                     "learner_hash": digest(hp)}
    if mode == "smoke":
        # Pipeline checks only (same reductions as tuning smoke); never part of the analysis.
        cfg["budget"]["env_steps"] = 64
        cfg["env"]["kwargs"]["max_steps"] = 32
        cfg["eval"].update(n_points=2, every_steps=32, periodic_episodes=1, final_episodes=1)
        cfg["checkpoint"]["every_steps"] = 32
        cfg["logging"]["train_log_every_steps"] = 16
        h = cfg["algo"]["hyperparams"]
        if env == "maintenance":
            h.update(n_steps=32, batch_size=16, n_epochs=1)
        elif algo in ("dqn", "dqn_ts"):
            h.update(buffer_size=128, learning_starts=32, batch_size=8, target_update_interval=32, train_freq=8)
        else:
            h.update(replay_frames=128, learning_starts=32, batch_size=2, target_update=32, train_freq=8)
            if algo == "drqn":
                h.update(train_len=8, burn_in=4, batch_size=1)
    base.validate_config(cfg)
    ev = cfg["eval"]
    periodic = eval_episode_seeds(ev["periodic_seed_block"], ev["periodic_episodes"])
    final = eval_episode_seeds(ev["final_seed_block"], ev["final_episodes"])
    if not set(periodic).isdisjoint(final):
        raise ValueError("periodic and final evaluation seeds overlap")
    return cfg


# --------------------------------------------------------------------------- run order
def role_queues():
    """role -> ordered list of (round, level, seed, env, algo)."""
    moved = set(MOVED_TO_C)
    queues = {}
    for role in ROLES:
        q = []
        for rnd, (level, seed) in enumerate(ROLE_SETS[role], start=1):
            for env, algo in CELL_ORDER:
                if (level, seed, env, algo) not in moved:
                    q.append((rnd, level, seed, env, algo))
        queues[role] = q
    last_round = len(ROLE_SETS["a"])
    queues["c"] += [(last_round, level, seed, env, algo) for level, seed, env, algo in MOVED_TO_C]
    return queues


def build_plan(selected_path=SELECTED_PATH):
    sel = load_selected(selected_path)
    runs = []
    for role, queue in role_queues().items():
        for slot, (rnd, level, seed, env, algo) in enumerate(queue):
            learner = sel["cells"][f"{env}/{algo}"]
            cfg = resolve_config(env, algo, level, seed, learner["hyperparams"], learner["candidate"])
            runs.append({"role": role, "slot": slot, "round": rnd, "env": env, "algo": algo, "level": level,
                         "seed": seed, "device": DEVICE_POLICY[env], "candidate": learner["candidate"],
                         "config_hash": base.config_hash(cfg), "config": cfg})
    plan = {
        "schema": SCHEMA, "protocol_version": "v3", "phase": PHASE, "decision": "D11",
        "protocol": "run_scripts/stress/STRESS_PROTOCOL.md",
        "tuning_plan_hash": sel["tuning_plan_hash"],
        "selected_learners": {cell: {"candidate": v["candidate"], "learner_hash": digest(v["hyperparams"])}
                              for cell, v in sel["cells"].items()},
        "levels": {env: {lv: base._jsonable(STRESS_LEVEL_OVERRIDES[env][lv]) for lv in STRESS_LEVELS}
                   for env in STRESS_LEVEL_OVERRIDES},
        "training_seeds": {"default": list(DEFAULT_SEEDS), "other_levels": list(STRESS_SEEDS)},
        "eval_seed_blocks": {kind: {"periodic": r[1], "final": r[2]} for kind, r in SEED_RULES.items()},
        "threads": THREADS, "device_policy": DEVICE_POLICY,
        "role_sets": {r: [list(s) for s in v] for r, v in ROLE_SETS.items()},
        "moved_to_c": [list(m) for m in MOVED_TO_C],
        "run_count": len(runs), "role_counts": {r: sum(x["role"] == r for x in runs) for r in ROLES},
        "runs": runs,
    }
    plan["plan_hash"] = digest(plan)
    return plan


# --------------------------------------------------------------------------- checks / lookup
def json_diff(a, b, path="", out=None, limit=50):
    """Paths where two JSON values differ (exact comparison, types included)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'missing in file' if k not in a else 'not expected'}")
            else:
                json_diff(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} in file vs {len(b)} expected")
        for i, (x, y) in enumerate(zip(a, b)):
            json_diff(x, y, f"{path}[{i}]", out, limit)
    elif type(a) is not type(b) or a != b:
        out.append(f"{path}: {a!r} in file vs {b!r} expected")
    return out


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = build_plan()
    if plan != expected:
        diff = json_diff(plan, expected)
        raise ValueError("stress plan differs from the frozen generator:\n  " + "\n  ".join(diff))
    return plan


def find_run(plan, env, algo, level, seed):
    match = [r for r in plan["runs"] if (r["env"], r["algo"], r["level"], r["seed"]) == (env, algo, level, int(seed))]
    if len(match) != 1:
        raise ValueError(f"{env}/{algo}/{level}/s{seed} is not a registered stress run")
    return match[0]


def run_config(plan, env, algo, level, seed, mode="full"):
    """The plan entry and the config to execute (smoke derives from the same tuned learner)."""
    entry = find_run(plan, env, algo, level, seed)
    if mode == "full":
        return entry, copy.deepcopy(entry["config"])
    sel = load_selected()["cells"][f"{env}/{algo}"]
    if digest(sel["hyperparams"]) != plan["selected_learners"][f"{env}/{algo}"]["learner_hash"]:
        raise ValueError("selected learner changed since the plan was frozen")
    return entry, resolve_config(env, algo, level, seed, sel["hyperparams"], sel["candidate"], "smoke")
