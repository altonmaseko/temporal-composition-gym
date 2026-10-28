"""Step 5: aggregate behaviour rollouts and build the machine-readable summary.

Reads out/reconcile.json, out/learning.json, out/behaviour_<env>.json, out/suspects.json and
writes out/behaviour_tables.csv, figures/behaviour_greedy_vs_explore.png and summary.json.

    python -B analysis/baseline_v2/05_summary.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import CELLS, COLOR, FAMILY, FIG, HERE, INK, INK2, LABEL, OUT, read_json, style_axes, write_json

rec = read_json(os.path.join(OUT, "reconcile.json"))
learn = read_json(os.path.join(OUT, "learning.json"))
sus = read_json(os.path.join(OUT, "suspects.json"))
hyp = read_json(os.path.join(HERE, "hypotheses.json"))

# ------------------------------------------------------------------ behaviour tables
rows = []
for env in CELLS:
    path = os.path.join(OUT, f"behaviour_{env}.json")
    if not os.path.exists(path):
        continue
    eps = pd.DataFrame([e for e in read_json(path)["episodes"] if not e.get("missing")])
    eps = eps.drop(columns=[c for c in ("trace",) if c in eps])
    for (algo, frac, mode), g in eps.groupby(["algo", "frac", "mode"]):
        r = {"env": env, "algo": algo, "snapshot_frac": frac, "mode": mode, "episodes": len(g),
             "success": g["success"].mean(), "return": g["return"].mean(), "disc_return": g["disc_return"].mean(),
             "progress": g["progress"].mean(), "frac_progress_pos": (g["progress"] > 0).mean(),
             "length": g["length"].mean(), "causes": json.dumps(g["cause"].value_counts(normalize=True).round(3).to_dict())}
        if "q0_max" in g and g["q0_max"].notna().any():
            r["q0_max_mean"] = g["q0_max"].mean()
            r["q0_minus_disc_return"] = (g["q0_max"] - g["disc_return"]).mean()
        if env == "delivery":
            r.update(dominant_action_frac=g["dominant_action_frac"].mean(), unique_cells=g["unique_cells"].mean(),
                     top_cell_frac=g["top_cell_frac"].mean(), pickups=g["pickups"].mean(),
                     frac_exact_state_cycle=g["cycle_len"].notna().mean(),
                     median_first_repeat_step=g["first_state_repeat_step"].median(),
                     cycle_len_mode=g["cycle_len"].mode().iloc[0] if g["cycle_len"].notna().any() else None)
        elif env == "colour":
            r.update(dominant_action_frac=g["dominant_action_frac"].mean(), wall_touches=g["wall_touches"].mean(),
                     frac_any_wall_touch=(g["wall_touches"] > 0).mean(), unique_nav_positions=g["unique_nav_positions"].mean(),
                     top_position_frac=g["top_position_frac"].mean(), max_disp_blocks=g["max_disp_blocks"].mean(),
                     first_nav_maxq=g["first_nav_maxq"].mean())
        else:
            for k in ("boundary_frac", "offpath_frac", "stages", "codes", "service_attempts", "mean_speed_cells_s",
                      "frac_slow", "thrust_mean", "thrust_pos_frac", "brake_frac", "torque_mean", "angle_abs_mean",
                      "y_range", "x_range"):
                r[k] = g[k].mean()
        rows.append(r)
beh = pd.DataFrame(rows)
beh.to_csv(os.path.join(OUT, "behaviour_tables.csv"), index=False)

# figure: greedy vs exploratory, final snapshot
if len(beh):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, env in zip(axes, CELLS):
        b = beh[(beh.env == env) & (beh.snapshot_frac == 1.0)]
        algos = [a for a in CELLS[env] if a in set(b.algo)]
        metric = "progress"
        for i, algo in enumerate(algos):
            c = COLOR[FAMILY[algo]]
            g = b[(b.algo == algo) & (b["mode"] == "greedy")][metric]
            e = b[(b.algo == algo) & (b["mode"] == "explore")][metric]
            ax.bar(i - 0.18, g.iloc[0] if len(g) else np.nan, width=0.34, color=c, alpha=0.45, edgecolor="white")
            ax.bar(i + 0.18, e.iloc[0] if len(e) else np.nan, width=0.34, color=c, edgecolor="white")
        ax.set_xticks(range(len(algos)), [LABEL[a].replace(" (control)", "\n(control)") for a in algos], fontsize=8)
        ax.set_title(f"{env}: progress per episode", fontsize=10, color=INK, loc="left")
        style_axes(ax)
    axes[0].text(0.02, 0.95, "pale = greedy (eval policy)\nsolid = exploratory (eps 0.05 / stochastic)",
                 transform=axes[0].transAxes, fontsize=7, color=INK2, va="top")
    fig.suptitle("Final snapshots, 5 seeds x first periodic-eval seeds: greedy vs exploratory behaviour", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "behaviour_greedy_vs_explore.png"), dpi=110)
    plt.close(fig)

# ------------------------------------------------------------------ summary.json
cells = {}
for name, c in learn["cells"].items():
    a = c["aggregate"]
    env, algo = name.split("/")
    b = beh[(beh.env == env) & (beh.algo == algo) & (beh.snapshot_frac == 1.0)] if len(beh) else pd.DataFrame()
    bg = b[b["mode"] == "greedy"].to_dict("records")
    be = b[b["mode"] == "explore"].to_dict("records")
    cells[name] = {
        "label": c["label"], "family": c["family"], "budget_env_steps": c["budget"], "seeds": 5, "runs_complete": 5,
        "final_eval": {k: a[f"final_{k}"] for k in ("success", "return", "norm_return", "progress", "length")},
        "final_causes_pooled": a["final_causes_pooled"],
        "best_periodic_success": a["best_success"], "auc_success": a["auc_success"], "auc_return": a["auc_return"],
        "auc_norm_return": a["auc_norm_return"],
        "train_successes_total": a["train_successes_total"], "train_episodes_total": a["train_episodes_total"],
        "late_train_vs_eval_mean": {k: float(np.mean([c["seeds"][s]["late_train_vs_eval"][k] for s in c["seeds"]]))
                                    for k in c["seeds"]["1"]["late_train_vs_eval"]},
        "behaviour_final_greedy": bg[0] if bg else None, "behaviour_final_explore": be[0] if be else None,
        "throughput_env_steps_per_s": a["steps_per_s"], "wall_clock_h_per_run": a["wall_time_h"],
        "failure_hypotheses": hyp["cells"].get(name, []),
    }

summary = {
    "protocol_version": "v2", "phase": "baseline analysis (Prompt 1)",
    "inputs": {"results": "run_scripts/baseline/results/v2/full (read-only)", "manifest": "manifest.json, 55/55 complete",
               "references": "run_scripts/baseline/validation/task_validation_v2.json"},
    "reconciliation": {"manifest": rec["manifest"]["summary_line"], "checks": rec["all"], "problems": rec["problems"],
                       "code_revision": list(rec["git_revisions"]),
                       "code_identical_to_validated_7d94a93": all(v["code_diff_vs_validated"] == "(none)"
                                                                 for v in rec["git_revisions"].values()),
                       "resumed_runs": rec["resumed_runs"], "failed_attempts": sum(b["nonzero_exit"] for b in rec["batches"].values()),
                       "machines": rec["machines"], "python": rec["python_versions"], "packages": rec["packages"]},
    "references": {env: {"oracle_success": v["oracle"]["success_rate"], "oracle_return": v["oracle"]["return_mean"],
                         "random_success": v["random"]["success_rate"], "random_return": v["random"]["return_mean"]}
                   for env, v in learn["references"].items()},
    "cells": cells,
    "suspect_checks": {k: v for k, v in sus.items()},
    "defects": hyp["defects"],
    "invalidates_baseline_results": hyp["invalidates_baseline_results"],
    "diagnostic_seed_ranges_used": hyp["diagnostic_seed_ranges_used"],
}
write_json(os.path.join(HERE, "summary.json"), summary)
print(beh.drop(columns=["causes"]).round(3).to_string(index=False))
print("wrote summary.json")
