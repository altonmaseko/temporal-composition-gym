"""Shared helpers for the protocol-v2 baseline analysis (read-only access to results).

Nothing here writes under run_scripts/baseline/results or run_scripts/baseline/validation.
All outputs go to analysis/baseline_v2/out and analysis/baseline_v2/figures.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

RESULTS = os.path.join(REPO, "run_scripts", "baseline", "results")
FULL = os.path.join(RESULTS, "v2", "full")
VALIDATION = os.path.join(REPO, "run_scripts", "baseline", "validation")
PLAN_JSON = os.path.join(REPO, "run_scripts", "baseline", "baseline_plan.json")
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

VALIDATED_REV = "7d94a93"
ENVS = ("delivery", "colour", "maintenance")
CELLS = {
    "delivery": ["dqn", "drqn", "qrm", "dqn_ts"],
    "colour": ["dqn", "drqn", "qrm_cue", "dqn_ts"],
    "maintenance": ["ppo", "recurrent_ppo", "ppo_rm"],
}
FAMILY = {"dqn": "memoryless", "ppo": "memoryless", "drqn": "implicit", "recurrent_ppo": "implicit",
          "qrm": "explicit", "qrm_cue": "explicit", "ppo_rm": "explicit", "dqn_ts": "control"}
LABEL = {"dqn": "DQN", "ppo": "PPO", "drqn": "DRQN", "recurrent_ppo": "Recurrent PPO", "qrm": "QRM",
         "qrm_cue": "QRM-Cue", "ppo_rm": "PPO-RM", "dqn_ts": "DQN-TS (control)"}
# Categorical slots 1-4 of the dataviz reference palette, fixed order by family.
COLOR = {"memoryless": "#2a78d6", "implicit": "#eb6834", "explicit": "#1baf7a", "control": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
SEEDS = (1, 2, 3, 4, 5)
SNAPSHOT_EXT = {"dqn": "zip", "dqn_ts": "zip", "ppo": "zip", "ppo_rm": "zip", "recurrent_ppo": "zip",
                "drqn": "pt", "qrm": "pt", "qrm_cue": "pt"}

# Diagnostic seed ranges proposed for this analysis (outside every reserved range).
DIAG_TRAIN_SEEDS = range(401, 500)
DIAG_EPISODE_SEED_BASE = 6_000_000


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, default=_np_default)
        f.write("\n")


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def run_dirs():
    """{(env, algo, seed): run_dir} for every run directory under results/v2/full."""
    out = {}
    for ident in glob.glob(os.path.join(FULL, "*", "*", "seed*__*", "identity.json")):
        d = os.path.dirname(ident)
        i = read_json(ident)
        key = (i["env"], i["algo"], int(i["seed"]))
        if key in out:
            raise RuntimeError(f"two run directories for {key}")
        out[key] = d
    return out


def references():
    """Oracle and uniform-random references from task_validation_v2.json."""
    tv = read_json(os.path.join(VALIDATION, "task_validation_v2.json"))
    out = {}
    for env, v in tv["environments"].items():
        out[env] = {"oracle": v["oracle"], "random": v["random"], "horizon": v.get("horizon")}
    return out


def normalise(ret, env, refs):
    r, o = refs[env]["random"]["return_mean"], refs[env]["oracle"]["return_mean"]
    return (np.asarray(ret, dtype=float) - r) / (o - r)


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    if hi <= lo:
        return float(np.mean(x))
    return float(np.mean(x[lo:hi]))


def bootstrap_ci(x, stat=np.mean, n=2000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def auc(steps, values, budget):
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    return float(np.trapezoid(values, steps) / budget)
