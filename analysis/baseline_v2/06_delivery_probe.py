"""Step 4b: why does a frozen delivery snapshot with epsilon 0.05 deliver far less than the
training episodes logged at the same epsilon? (read-only, no learning)

For each delivery cell and seed, the final snapshot is rolled out with epsilon in
{0, 0.05, 0.2, 1.0} on 20 periodic-eval seeds and on 20 diagnostic seeds (6,000,000+), and
compared with the training episodes logged in the last 5% of the run. If the gap is a
train/eval pipeline mismatch it should persist on every seed set and epsilon; if it comes
from deterministic loops of a frozen policy, more action noise should close it.

    python -B analysis/baseline_v2/06_delivery_probe.py
"""
import os

import numpy as np
import pandas as pd
import torch

from common import CELLS, DIAG_EPISODE_SEED_BASE, OUT, SEEDS, read_json, run_dirs, write_json
from agents.common.runner import build_env
from agents.common.seeding import EVAL_EPISODE_SEED_BASE

import importlib.util
spec = importlib.util.spec_from_file_location("beh", os.path.join(os.path.dirname(__file__), "03_behaviour.py"))
beh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(beh)

torch.set_num_threads(2)
RUNS = run_dirs()
N = 20
seed_sets = {"periodic": [EVAL_EPISODE_SEED_BASE["periodic"] + i for i in range(N)],
             "diagnostic": [DIAG_EPISODE_SEED_BASE + 1000 + i for i in range(N)]}
rows, train_rows = [], []
for algo in CELLS["delivery"]:
    for s in SEEDS:
        d = RUNS[("delivery", algo, s)]
        cfg = read_json(os.path.join(d, "config.resolved.json"))
        env = build_env(cfg)
        pol = beh.load_policy(algo, beh.snapshot_path(d, algo, 1_000_000), cfg, env)
        for set_name, eps_seeds in seed_sets.items():
            for eps in (0.0, 0.05, 0.2, 1.0):
                rng = np.random.default_rng([DIAG_EPISODE_SEED_BASE, s, int(eps * 100), 11])
                for es in eps_seeds:
                    r = beh.rollout("delivery", algo, env, pol, es, "explore" if eps > 0 else "greedy", rng, eps=eps)
                    rows.append({"algo": algo, "seed": s, "seed_set": set_name, "eps": eps, "progress": r["progress"],
                                 "return": r["return"], "length": r["length"], "cause": r["cause"],
                                 "success": r["success"], "pickups": r["pickups"]})
        env.close()
        ep = pd.read_csv(os.path.join(d, "train_episodes.csv"))
        late = ep[ep["global_step"] > 950_000]
        train_rows.append({"algo": algo, "seed": s, "episodes": len(late), "progress": late["progress"].mean(),
                           "return": late["return_raw"].mean(), "length": late["length"].mean(),
                           "lava": (late["cause"] == "lava").mean(), "timeout": (late["cause"] == "timeout").mean(),
                           "success": late["success"].mean()})
        print(algo, s, flush=True)

df, tr = pd.DataFrame(rows), pd.DataFrame(train_rows)
tab = df.groupby(["algo", "seed_set", "eps"]).agg(progress=("progress", "mean"), ret=("return", "mean"),
                                                   length=("length", "mean"), success=("success", "mean"),
                                                   lava=("cause", lambda c: (c == "lava").mean()),
                                                   timeout=("cause", lambda c: (c == "timeout").mean())).reset_index()
ttab = tr.groupby("algo").mean(numeric_only=True).reset_index()
write_json(os.path.join(OUT, "delivery_probe.json"), {"episodes_per_cell": N * 5, "rollouts": tab.to_dict("records"),
                                                      "training_last_5pct": ttab.to_dict("records")})
pd.set_option("display.width", 200)
print(tab.round(3).to_string(index=False))
print(ttab.round(3).to_string(index=False))
