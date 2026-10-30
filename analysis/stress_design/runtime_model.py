"""Stress-design probe: full-budget wall-clock per cell from the verified tuning records.

Read-only over run_scripts/tuning/results/v3/full and run_scripts/baseline/results/v2/full.

Evaluation is a fixed cost that does not scale with the training budget (greedy and noisy,
21 points, mostly timeout-length episodes), so "tuning time x 4" overstates it. Per run:

    k            = final-evaluation seconds / final-evaluation env steps   (measured directly:
                   finished_utc - started_utc - last resource wall time; greedy + noisy)
    eval_s       = k * (periodic eval steps, greedy + noisy) + final-evaluation seconds
    train_s      = total_s - eval_s                     (includes setup and checkpointing)
    T_full       = 4 * train_s + eval_s                 (same hardware and thread count)

Eval steps come from the eval_*.csv lengths of the selected candidate's tuning runs; the
full-budget runs use the same episode counts, so eval_s carries over unchanged. The v2
full-run wall-clock (RTX 3090, 4 threads, greedy evaluation only) is shown for comparison.
Output: out/runtime_model.json
"""
import csv
import glob
import json
import os
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TUNE = os.path.join(ROOT, "run_scripts", "tuning", "results", "v3", "full")
V2 = os.path.join(ROOT, "run_scripts", "baseline", "results", "v2", "full")
OUT = os.path.join(ROOT, "analysis", "stress_design", "out")
FULL = {"delivery": 1_000_000, "colour": 3_000_000, "maintenance": 2_048_000}


def ts(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_record(d):
    md = json.load(open(os.path.join(d, "metadata.json"), encoding="utf-8"))
    st = json.load(open(os.path.join(d, "status.json"), encoding="utf-8"))
    res = rows(os.path.join(d, "resources.csv"))
    gpu = (md["device"].get("gpu_name") or "cpu") if md.get("device") else "cpu"
    eval_len = {}
    for name in ("eval_periodic.csv", "eval_periodic_noisy.csv"):
        for r in rows(os.path.join(d, name)):
            i = int(r["eval_index"])
            eval_len[i] = eval_len.get(i, 0) + int(r["length"])
    final_len = sum(int(r["length"]) for n in ("eval_final.csv", "eval_final_noisy.csv")
                    for r in rows(os.path.join(d, n)))
    walls = [float(r["wall_time_s"]) for r in res]
    total = ts(st["finished_utc"]) - ts(st["started_utc"])
    return {"dir": os.path.relpath(d, ROOT), "gpu": gpu, "threads": md.get("threads", {}).get("requested"),
            "cpu": md["hardware"].get("processor"), "resumes": len(st.get("resumes", [])),
            "intervals": [walls[i] - walls[i - 1] for i in range(1, len(walls))],
            "eval_steps": [eval_len.get(i, 0) for i in range(len(walls))],
            "final_eval_steps": final_len, "final_eval_s": total - walls[-1], "total_s": total}


def main():
    sel = json.load(open(os.path.join(TUNE, "selected_configs.json"), encoding="utf-8"))["selected"]
    out = {"method": __doc__.strip().splitlines()[0], "cells": {}}
    for cell, info in sel.items():
        env, algo = cell.split("/")
        cand = f"t{int(info['candidate']):02d}"
        dirs = sorted(glob.glob(os.path.join(TUNE, env, algo, cand, "s*", "a*")))
        recs = [run_record(d) for d in dirs if os.path.exists(os.path.join(d, "status.json"))]
        recs = [r for r in recs if r["resumes"] == 0]
        per = []
        for r in recs:
            k = r["final_eval_s"] / r["final_eval_steps"]
            ev = k * sum(r["eval_steps"]) + r["final_eval_s"]
            per.append((k, ev, r["total_s"] - ev))
        k = float(np.mean([p[0] for p in per]))
        eval_full = float(np.mean([p[1] for p in per]))
        train_full = 4 * float(np.mean([p[2] for p in per]))
        e_periodic = float(np.mean([sum(r["eval_steps"]) for r in recs]))
        t_final = float(np.mean([r["final_eval_s"] for r in recs]))
        tune_total = float(np.mean([r["total_s"] for r in recs]))
        v2 = []
        for d in glob.glob(os.path.join(V2, env, algo, "*")):
            if os.path.exists(os.path.join(d, "status.json")):
                st = json.load(open(os.path.join(d, "status.json"), encoding="utf-8"))
                v2.append(ts(st["finished_utc"]) - ts(st["started_utc"]))
        out["cells"][cell] = {
            "candidate": cand, "runs_used": [r["dir"] for r in recs], "gpus": sorted({r["gpu"] for r in recs}),
            "threads": sorted({r["threads"] for r in recs}),
            "eval_s_per_step": k, "eval_steps_per_s": 1 / k,
            "eval_steps_periodic_total": e_periodic, "final_eval_s": t_final,
            "tuning_total_h_mean": tune_total / 3600,
            "tuning_train_h": train_full / 4 / 3600, "tuning_eval_h": eval_full / 3600,
            "projected_full_train_h": train_full / 3600, "projected_full_eval_h": eval_full / 3600,
            "projected_full_total_h": (train_full + eval_full) / 3600,
            "naive_tuning_x4_h": 4 * tune_total / 3600,
            "v2_full_h_mean": float(np.mean(v2)) / 3600 if v2 else None,
        }
        o = out["cells"][cell]
        print(f"{cell:28s} {cand} gpus={o['gpus']} train={o['projected_full_train_h']:.2f}h "
              f"eval={o['projected_full_eval_h']:.2f}h total={o['projected_full_total_h']:.2f}h "
              f"(x4={o['naive_tuning_x4_h']:.2f}h, v2={o['v2_full_h_mean'] or float('nan'):.2f}h)")
    tot = sum(v["projected_full_total_h"] for v in out["cells"].values())
    out["one_set_total_h"] = tot
    print(f"one set of 11 cells: {tot:.1f} machine-hours")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "runtime_model.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
