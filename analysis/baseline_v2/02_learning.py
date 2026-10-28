"""Step 2: learning diagnostics per environment/algorithm cell from the local CSVs (read-only).

Writes out/learning.json, out/final_by_run.csv and figures/*.png.

    D:\\miniconda3\\envs\\temporal_gym\\python.exe -B analysis/baseline_v2/02_learning.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (CELLS, COLOR, FAMILY, FIG, INK, INK2, LABEL, OUT, SEEDS, auc, bootstrap_ci, iqm, normalise,
                    read_json, references, run_dirs, style_axes, write_json)

REFS = references()
RUNS = run_dirs()
CAUSE_ORDER = ["success", "lava", "damage_death", "premature_goal", "wrong_wall_death",
               "uncollected_codes_at_exit", "held_codes_at_final_exit", "timeout"]
CAUSE_COLOR = {"success": "#008300", "lava": "#e34948", "damage_death": "#e87ba4", "premature_goal": "#eda100",
               "wrong_wall_death": "#6250d6", "uncollected_codes_at_exit": "#eb6834",
               "held_codes_at_final_exit": "#6250d6", "timeout": "#b5b4ae"}
LEARNING_STARTS = {}


def load(key):
    d = RUNS[key]
    f = lambda n: pd.read_csv(os.path.join(d, n))  # noqa: E731
    return {"cfg": read_json(os.path.join(d, "config.resolved.json")), "pts": f("eval_points.csv"),
            "per": f("eval_periodic.csv"), "fin": f("eval_final.csv"), "ep": f("train_episodes.csv"),
            "upd": f("train_updates.csv"), "res": f("resources.csv"), "fs": read_json(os.path.join(d, "final_summary.json"))}


def causes_frac(series):
    vc = series.value_counts(normalize=True)
    return {c: float(vc.get(c, 0.0)) for c in CAUSE_ORDER if vc.get(c, 0.0) > 0}


def extras(upd):
    rows = []
    for _, r in upd.iterrows():
        e = json.loads(r["extra_json"]) if isinstance(r["extra_json"], str) else {}
        e["global_step"] = r["global_step"]
        rows.append(e)
    return pd.DataFrame(rows)


def at_fracs(df, col, budget, fracs=(0.1, 0.2, 0.5, 1.0)):
    out = {}
    d = df.dropna(subset=[col])
    for fr in fracs:
        sub = d[d["global_step"] <= fr * budget + 1]
        if len(sub):
            out[f"{int(fr * 100)}%"] = float(sub[col].iloc[-1])
    return out


data = {k: load(k) for k in sorted(RUNS)}
summary = {"references": REFS, "cells": {}}
final_rows = []

for env, algos in CELLS.items():
    budget = data[(env, algos[0], 1)]["cfg"]["budget"]["env_steps"]
    horizon = data[(env, algos[0], 1)]["cfg"]["env"]["kwargs"]["max_steps"]
    for algo in algos:
        cell = {"label": LABEL[algo], "family": FAMILY[algo], "budget": budget, "seeds": {}}
        hp = data[(env, algo, 1)]["cfg"]["algo"]["hyperparams"]
        ls = hp.get("learning_starts")
        LEARNING_STARTS[(env, algo)] = ls
        curves = []
        for s in SEEDS:
            D = data[(env, algo, s)]
            pts, fin, ep, upd, res = D["pts"], D["fin"], D["ep"], D["upd"], D["res"]
            rs = {}
            # ---- final evaluation
            rs["final"] = {"success": float(fin["success"].mean()), "return": float(fin["return_raw"].mean()),
                           "norm_return": float(normalise(fin["return_raw"].mean(), env, REFS)),
                           "progress": float(fin["progress"].mean()), "length": float(fin["length"].mean()),
                           "frac_progress_pos": float((fin["progress"] > 0).mean()),
                           "max_progress": int(fin["progress"].max()), "causes": causes_frac(fin["cause"])}
            # ---- periodic evaluation
            rs["periodic"] = {
                "best_success": float(pts["success_rate"].max()),
                "best_success_step": int(pts.loc[pts["success_rate"].idxmax(), "global_step"]),
                "best_return": float(pts["return_mean"].max()),
                "auc_success": auc(pts["global_step"], pts["success_rate"], budget),
                "auc_return": auc(pts["global_step"], pts["return_mean"], budget),
                "auc_norm_return": auc(pts["global_step"], normalise(pts["return_mean"], env, REFS), budget),
                "step0_return": float(pts["return_mean"].iloc[0]), "step0_success": float(pts["success_rate"].iloc[0]),
                "n_points_with_any_success": int((pts["success_rate"] > 0).sum()),
                "max_progress_any_episode": int(D["per"]["progress"].max()),
            }
            # ---- training episodes
            ep = ep.copy()
            ep["window"] = np.minimum((ep["global_step"] - 1) // (budget // 10), 9)
            win = ep.groupby("window").agg(n=("return_raw", "size"), ret=("return_raw", "mean"),
                                           success=("success", "mean"), progress=("progress", "mean"),
                                           prog_pos=("progress", lambda x: float((x > 0).mean())),
                                           length=("length", "mean"))
            pre = ep[ep["global_step"] <= ls] if ls else ep.iloc[:0]
            succ = ep[ep["success"] == 1]
            rs["train"] = {
                "episodes": int(len(ep)), "mean_length": float(ep["length"].mean()),
                "successes": int(len(succ)), "first_success_step": int(succ["global_step"].iloc[0]) if len(succ) else None,
                "max_progress": int(ep["progress"].max()),
                "progress_hist": {int(k): int(v) for k, v in ep["progress"].value_counts().sort_index().items()},
                "causes": causes_frac(ep["cause"]),
                "windows_10pct": {int(w): {k: float(v) for k, v in row.items()} for w, row in win.iterrows()},
                "causes_first_10pct": causes_frac(ep[ep["window"] == 0]["cause"]),
                "causes_last_10pct": causes_frac(ep[ep["window"] == 9]["cause"]),
                "learning_starts": ls,
                "pre_learning": None if not ls else {
                    "episodes": int(len(pre)), "with_progress": int((pre["progress"] > 0).sum()),
                    "successes": int(pre["success"].sum()), "max_return": float(pre["return_raw"].max()) if len(pre) else None,
                    "positive_reward_episodes": int((pre["return_raw"] > 0).sum()),
                    "mean_return": float(pre["return_raw"].mean()) if len(pre) else None,
                    "causes": causes_frac(pre["cause"])},
            }
            # late training (epsilon at its floor / PPO stochastic) vs eval at the same steps
            late = ep[ep["global_step"] > 0.8 * budget]
            late_eval = D["per"][D["per"]["global_step"] > 0.8 * budget]
            rs["late_train_vs_eval"] = {
                "train_return": float(late["return_raw"].mean()), "eval_return": float(late_eval["return_raw"].mean()),
                "train_progress": float(late["progress"].mean()), "eval_progress": float(late_eval["progress"].mean()),
                "train_success": float(late["success"].mean()), "eval_success": float(late_eval["success"].mean()),
                "train_length": float(late["length"].mean()), "eval_length": float(late_eval["length"].mean()),
                "train_timeout_frac": float((late["cause"] == "timeout").mean()),
                "eval_timeout_frac": float((late_eval["cause"] == "timeout").mean()),
            }
            # ---- updates
            x = extras(upd)
            u = {"loss": at_fracs(upd, "loss", budget), "q_mean": at_fracs(upd, "q_mean", budget),
                 "grad_norm": at_fracs(upd, "grad_norm", budget), "epsilon": at_fracs(upd, "epsilon", budget)}
            for k in ("train/approx_kl", "train/clip_fraction", "train/std", "train/entropy_loss", "train/value_loss",
                      "train/explained_variance", "train/policy_gradient_loss", "valid_heads_per_transition",
                      "trained_transitions"):
                if k in x.columns:
                    u[k] = at_fracs(x, k, budget)
                    u[k + "__mean"] = float(x[k].mean())
            u["q_mean_max"] = float(upd["q_mean"].max()) if upd["q_mean"].notna().any() else None
            u["loss_nonfinite_rows"] = int((~np.isfinite(upd["loss"].dropna())).sum())
            rs["updates"] = u
            # ---- resources
            rs["resources"] = {"steps_per_s_mean": float(res["env_steps_per_s"].dropna().mean()),
                               "wall_time_h": float(res["wall_time_s"].iloc[-1] / 3600),
                               "rss_gb_max": float(res["rss_gb"].max()),
                               "gpu_mem_max_gb": float(res["gpu_mem_max_gb"].max()) if "gpu_mem_max_gb" in res else None}
            cell["seeds"][s] = rs
            final_rows.append({"env": env, "algo": algo, "seed": s, **{f"final_{k}": v for k, v in rs["final"].items()
                                                                          if k != "causes"},
                               "final_causes": json.dumps(rs["final"]["causes"]),
                               **{f"periodic_{k}": v for k, v in rs["periodic"].items()},
                               "train_successes": rs["train"]["successes"], "train_episodes": rs["train"]["episodes"]})
            curves.append(pts.assign(seed=s))
        # ---- cell aggregates (5 seeds)
        agg = {}
        for key in ("success", "return", "norm_return", "progress", "length"):
            v = [cell["seeds"][s]["final"][key] for s in SEEDS]
            agg[f"final_{key}"] = {"mean": float(np.mean(v)), "iqm": iqm(v), "min": float(np.min(v)),
                                   "max": float(np.max(v)), "ci95_mean": bootstrap_ci(v)}
        for key in ("best_success", "auc_success", "auc_return", "auc_norm_return"):
            v = [cell["seeds"][s]["periodic"][key] for s in SEEDS]
            agg[key] = {"mean": float(np.mean(v)), "iqm": iqm(v), "min": float(np.min(v)), "max": float(np.max(v))}
        allfin = pd.concat([data[(env, algo, s)]["fin"] for s in SEEDS])
        agg["final_causes_pooled"] = causes_frac(allfin["cause"])
        agg["train_successes_total"] = int(sum(cell["seeds"][s]["train"]["successes"] for s in SEEDS))
        agg["train_episodes_total"] = int(sum(cell["seeds"][s]["train"]["episodes"] for s in SEEDS))
        agg["steps_per_s"] = float(np.mean([cell["seeds"][s]["resources"]["steps_per_s_mean"] for s in SEEDS]))
        agg["wall_time_h"] = float(np.mean([cell["seeds"][s]["resources"]["wall_time_h"] for s in SEEDS]))
        cell["aggregate"] = agg
        cell["curve"] = (pd.concat(curves).groupby("global_step")[["success_rate", "return_mean", "progress_mean", "length_mean"]]
                         .mean().reset_index().to_dict("list"))
        summary["cells"][f"{env}/{algo}"] = cell

write_json(os.path.join(OUT, "learning.json"), summary)
pd.DataFrame(final_rows).to_csv(os.path.join(OUT, "final_by_run.csv"), index=False)


# ====================================================================== figures
def ref_lines(ax, env, metric):
    o, r = REFS[env]["oracle"], REFS[env]["random"]
    vals = {"success": (o["success_rate"], r["success_rate"]), "return": (o["return_mean"], r["return_mean"]),
            "length": (o["length_p50"], r["length_p50"])}.get(metric)
    if vals is None:
        return
    for v, name in zip(vals, ("oracle", "random")):
        ax.axhline(v, color=INK2, lw=1, ls="--" if name == "oracle" else ":")
        ax.annotate(name, (1.0, v), xycoords=("axes fraction", "data"), fontsize=7, color=INK2, ha="right", va="bottom")


def fig_eval(env):
    algos = CELLS[env]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    specs = [("success_rate", "Success rate (periodic eval)", "success"), ("return_mean", "Raw return", "return"),
             ("progress_mean", "Progress (deliveries / walls / services)", None), ("length_mean", "Episode length", "length")]
    for ax, (col, title, ref) in zip(axes.flat, specs):
        for algo in algos:
            c = COLOR[FAMILY[algo]]
            ys = []
            for s in SEEDS:
                p = data[(env, algo, s)]["pts"]
                ax.plot(p["global_step"] / 1e6, p[col], color=c, lw=0.6, alpha=0.35)
                ys.append(p[col].values)
            ax.plot(p["global_step"] / 1e6, np.mean(ys, axis=0), color=c, lw=2, label=LABEL[algo])
        ref_lines(ax, env, ref)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("environment steps (M)", fontsize=8, color=INK2)
        style_axes(ax)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{env}: periodic greedy evaluation (thin = seeds, thick = mean of 5)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"eval_curves_{env}.png"), dpi=110)
    plt.close(fig)


def fig_causes(env, source):
    algos = CELLS[env]
    fig, axes = plt.subplots(1, len(algos), figsize=(3.4 * len(algos), 3.2), sharey=True)
    for ax, algo in zip(axes, algos):
        if source == "eval":
            df = pd.concat([data[(env, algo, s)]["per"] for s in SEEDS])
            df = df.assign(x=df["global_step"])
        else:
            budget = data[(env, algo, 1)]["cfg"]["budget"]["env_steps"]
            df = pd.concat([data[(env, algo, s)]["ep"] for s in SEEDS])
            df = df.assign(x=(np.minimum((df["global_step"] - 1) // (budget // 20), 19) + 0.5) * (budget // 20))
        tab = pd.crosstab(df["x"], df["cause"], normalize="index")
        cols = [c for c in CAUSE_ORDER if c in tab.columns]
        ax.stackplot(tab.index / 1e6, [tab[c] for c in cols], colors=[CAUSE_COLOR[c] for c in cols],
                     labels=cols, edgecolor="white", linewidth=0.5)
        ax.set_title(LABEL[algo], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("steps (M)", fontsize=8, color=INK2)
        ax.set_ylim(0, 1)
        style_axes(ax)
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=len(labels))
    what = "periodic greedy evaluation" if source == "eval" else "training episodes (5% bins)"
    fig.suptitle(f"{env}: termination causes, {what}, 5 seeds pooled", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, f"causes_{source}_{env}.png"), dpi=110)
    plt.close(fig)


def fig_train(env):
    algos = CELLS[env]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for algo in algos:
        c = COLOR[FAMILY[algo]]
        budget = data[(env, algo, 1)]["cfg"]["budget"]["env_steps"]
        bins = budget // 50
        r, p, sp = [], [], []
        for s in SEEDS:
            ep = data[(env, algo, s)]["ep"]
            b = (ep["global_step"] - 1) // bins
            g = ep.groupby(b)
            r.append(g["return_raw"].mean().reindex(range(50)))
            p.append(g["progress"].mean().reindex(range(50)))
            sp.append(g["progress"].apply(lambda x: (x > 0).mean()).reindex(range(50)))
        x = (np.arange(50) + 0.5) * bins / 1e6
        for ax, arr in zip(axes, (r, p, sp)):
            ax.plot(x, np.nanmean(np.vstack(arr), axis=0), color=c, lw=2, label=LABEL[algo])
    titles = ["Training-episode raw return", "Training-episode progress (mean)", "Fraction of training episodes with progress > 0"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("environment steps (M)", fontsize=8, color=INK2)
        style_axes(ax)
    ref_lines(axes[0], env, "return")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{env}: training episodes (behaviour policy incl. exploration), mean of 5 seeds, 2% bins",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"train_episodes_{env}.png"), dpi=110)
    plt.close(fig)


def fig_updates(env):
    algos = CELLS[env]
    ppo = env == "maintenance"
    if ppo:
        keys = [("train/value_loss", "value loss"), ("train/explained_variance", "explained variance"),
                ("train/std", "policy action std"), ("train/approx_kl", "approx KL"),
                ("train/clip_fraction", "clip fraction"), ("train/policy_gradient_loss", "policy-gradient loss")]
    else:
        keys = [("loss", "TD loss (SB3 DQN: last update only)"), ("q_mean", "mean Q(s,a) of batch (custom agents)"),
                ("grad_norm", "grad norm before clip (custom agents)"), ("epsilon", "epsilon")]
    fig, axes = plt.subplots(2, 3 if ppo else 2, figsize=(13 if ppo else 10, 6.5))
    for algo in algos:
        c = COLOR[FAMILY[algo]]
        for ax, (k, t) in zip(axes.flat, keys):
            ys = []
            for s in SEEDS:
                upd = data[(env, algo, s)]["upd"]
                src = upd if k in upd.columns else extras(upd)
                if k not in src.columns:
                    continue
                d = src[["global_step", k]].dropna()
                ax.plot(d["global_step"] / 1e6, d[k], color=c, lw=0.6, alpha=0.4,
                        label=LABEL[algo] if s == 1 else None)
            ax.set_title(t, fontsize=9, color=INK, loc="left")
    for ax in axes.flat:
        ax.set_xlabel("environment steps (M)", fontsize=8, color=INK2)
        style_axes(ax)
        if ppo and ax.get_title() in ("value loss", "approx KL"):
            ax.set_yscale("log")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{env}: optimisation diagnostics from train_updates.csv (one line per seed)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"updates_{env}.png"), dpi=110)
    plt.close(fig)


def fig_final():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, env in zip(axes, CELLS):
        for i, algo in enumerate(CELLS[env]):
            v = [summary["cells"][f"{env}/{algo}"]["seeds"][s]["final"]["norm_return"] for s in SEEDS]
            c = COLOR[FAMILY[algo]]
            ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), v, s=36, color=c, edgecolor="white", linewidth=1, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [iqm(v)] * 2, color=INK, lw=2)
        ax.axhline(0, color=INK2, ls=":", lw=1)
        ax.axhline(1, color=INK2, ls="--", lw=1)
        ax.annotate("random", (1.0, 0), xycoords=("axes fraction", "data"), fontsize=7, color=INK2, ha="right", va="bottom")
        ax.annotate("oracle", (1.0, 1), xycoords=("axes fraction", "data"), fontsize=7, color=INK2, ha="right", va="top")
        ax.set_xticks(range(len(CELLS[env])), [LABEL[a].replace(" (control)", "\n(control)") for a in CELLS[env]], fontsize=8)
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"{env}", fontsize=10, color=INK, loc="left")
        style_axes(ax)
    axes[0].set_ylabel("normalised final return\n(R - R_random) / (R_oracle - R_random)", fontsize=8, color=INK2)
    fig.suptitle("Final evaluation (200 / 100 episodes per run): dots = seeds, bar = IQM. Success is 0 almost everywhere.",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "final_normalised_return.png"), dpi=110)
    plt.close(fig)


for env in CELLS:
    fig_eval(env)
    fig_causes(env, "eval")
    fig_causes(env, "train")
    fig_train(env)
    fig_updates(env)
fig_final()

# console summary
for name, cell in summary["cells"].items():
    a = cell["aggregate"]
    print(f"{name:28s} final succ {a['final_success']['mean']:.3f} ret {a['final_return']['mean']:9.2f} "
          f"norm {a['final_norm_return']['mean']:.3f} prog {a['final_progress']['mean']:.2f} len {a['final_length']['mean']:7.1f} "
          f"best {a['best_success']['max']:.2f} trainSucc {a['train_successes_total']}/{a['train_episodes_total']} "
          f"sps {a['steps_per_s']:.0f} causes {a['final_causes_pooled']}")
