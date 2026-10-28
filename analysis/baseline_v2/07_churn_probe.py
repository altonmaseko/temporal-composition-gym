"""Step 4c: does switching between snapshots (a crude proxy for the online agent's changing
weights, "policy churn") raise deliveries towards the training-episode level? (no learning)

Memoryless delivery agents (DQN, DQN-TS, QRM), final snapshot vs a mixture that picks one of
the 700k/800k/900k/1M snapshots uniformly every 4 steps; epsilon 0.05; 20 diagnostic episode
seeds (6,000,000+2000+i) per run.

    python -B analysis/baseline_v2/07_churn_probe.py
"""
import os

import numpy as np
import pandas as pd
import torch

from common import DIAG_EPISODE_SEED_BASE, OUT, SEEDS, read_json, run_dirs, write_json
from agents.common.runner import build_env

import importlib.util
spec = importlib.util.spec_from_file_location("beh", os.path.join(os.path.dirname(__file__), "03_behaviour.py"))
beh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(beh)
torch.set_num_threads(2)

RUNS = run_dirs()
STEPS = (700_000, 800_000, 900_000, 1_000_000)
rows = []
for algo in ("dqn", "dqn_ts", "qrm"):
    for s in SEEDS:
        d = RUNS[("delivery", algo, s)]
        cfg = read_json(os.path.join(d, "config.resolved.json"))
        env = build_env(cfg)
        pols = [beh.load_policy(algo, beh.snapshot_path(d, algo, st), cfg, env) for st in STEPS]
        for mode in ("final", "mixture"):
            rng = np.random.default_rng([DIAG_EPISODE_SEED_BASE, s, 21, mode == "mixture"])
            for i in range(20):
                obs, info = env.reset(seed=DIAG_EPISODE_SEED_BASE + 2000 + i)
                for p in pols:
                    p.reset(obs, info)
                t, cur = 0, pols[-1]
                while True:
                    if mode == "mixture" and t % 4 == 0:
                        cur = pols[int(rng.integers(len(pols)))]
                    a = int(np.argmax(cur.q(obs)))
                    if rng.random() < 0.05:
                        a = int(rng.integers(4))
                    obs, r, term, trunc, info = env.step(a)
                    for p in pols:
                        p.post_step(info)
                    t += 1
                    if term or trunc:
                        rec = info["episode_record"]
                        break
                rows.append({"algo": algo, "seed": s, "mode": mode, "progress": rec["progress"],
                             "return": rec["return_raw"], "cause": rec["cause"], "length": rec["length"]})
        env.close()
        print(algo, s, flush=True)
df = pd.DataFrame(rows)
tab = df.groupby(["algo", "mode"]).agg(progress=("progress", "mean"), ret=("return", "mean"), length=("length", "mean"),
                                       lava=("cause", lambda c: (c == "lava").mean()),
                                       timeout=("cause", lambda c: (c == "timeout").mean())).reset_index()
write_json(os.path.join(OUT, "churn_probe.json"), {"snapshots": STEPS, "epsilon": 0.05, "episodes_per_cell_mode": 100,
                                                   "table": tab.to_dict("records")})
print(tab.round(3).to_string(index=False))
