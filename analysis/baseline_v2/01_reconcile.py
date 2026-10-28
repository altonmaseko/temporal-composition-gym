"""Step 1: reconcile the 55 protocol-v2 baseline runs against the plan (read-only).

Checks: manifest re-run, config hashes vs baseline_plan.json, budgets and step accounting,
seed assignment and independence, code revision, devices, resumes and failed attempts,
gradient-update counts. Writes out/reconcile.json and out/manifest_recheck.json.

    D:\\miniconda3\\envs\\temporal_gym\\python.exe -B analysis/baseline_v2/01_reconcile.py
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from common import (CELLS, FULL, OUT, PLAN_JSON, REPO, SEEDS, VALIDATED_REV, read_json, run_dirs, write_json)
from agents.common import baseline_plan as plan
from agents.common.seeding import EVAL_EPISODE_SEED_BASE, TRAIN_SEED_RANGES, train_env_seed

PY = sys.executable
problems = []


def problem(msg):
    problems.append(msg)
    print("PROBLEM:", msg)


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


# ------------------------------------------------------------------ 1. manifest re-run
recheck_path = os.path.join(OUT, "manifest_recheck.json")
r = subprocess.run([PY, "-B", os.path.join(REPO, "run_scripts", "baseline", "check_manifest.py"), "--json", recheck_path],
                   cwd=REPO, capture_output=True, text=True)
manifest_stdout_tail = r.stdout.strip().splitlines()[-1]
recheck = read_json(recheck_path)
committed = read_json(os.path.join(FULL, "manifest.json"))
manifest = {
    "exit_code": r.returncode, "summary_line": manifest_stdout_tail, "counts": recheck["counts"],
    "all_complete": recheck["all_complete"], "unexpected_runs": len(recheck["unexpected_runs"]),
    "committed_counts": committed["counts"],
    "states_equal_committed": [(e["env"], e["algo"], e["seed"], e["state"]) for e in recheck["entries"]]
    == [(e["env"], e["algo"], e["seed"], e["state"]) for e in committed["entries"]],
    "attempts_per_planned_run": sorted({len(e["attempts"]) for e in recheck["entries"]}),
}
if r.returncode != 0 or not recheck["all_complete"]:
    problem(f"check_manifest exit {r.returncode}: {manifest_stdout_tail}")

# plan file up to date with the code?
pc = subprocess.run([PY, "-B", "-m", "agents.common.baseline_plan", "write", "--check"], cwd=REPO,
                    capture_output=True, text=True)
plan_check = {"exit_code": pc.returncode, "output": (pc.stdout + pc.stderr).strip()}
if pc.returncode != 0:
    problem("baseline_plan.json differs from the plan code")

plan_json = read_json(PLAN_JSON)
plan_hash = {(x["env"], x["algo"], x["seed"]): x["config_hash"] for x in plan_json["runs"]}
plan_cfg = plan_json["resolved_configs"]

# ------------------------------------------------------------------ 2. per-run checks
runs = run_dirs()
expected_keys = {(e, a, s) for e, algos in CELLS.items() for a in algos for s in SEEDS}
if set(runs) != expected_keys:
    problem(f"run set differs from plan: missing {sorted(expected_keys - set(runs))}, extra {sorted(set(runs) - expected_keys)}")
stray = [d for d in glob.glob(os.path.join(FULL, "[!_]*", "*", "*")) if os.path.isdir(d)
         and not os.path.exists(os.path.join(d, "identity.json"))]
if stray:
    problem(f"run-like directories without identity.json: {stray}")


def expected_updates(cfg):
    hp, b = cfg["algo"]["hyperparams"], cfg["budget"]["env_steps"]
    name = cfg["algo"]["name"]
    if name in ("dqn", "dqn_ts"):                  # SB3: train when num_timesteps > learning_starts, every train_freq
        return (b - hp["learning_starts"]) // hp["train_freq"]
    if name in ("drqn", "qrm", "qrm_cue"):         # custom: step >= learning_starts and step % train_freq == 0
        ls, f = hp["learning_starts"], hp["train_freq"]
        return b // f - (ls - 1) // f
    if name in ("ppo", "ppo_rm", "recurrent_ppo"):  # SB3 PPO counts n_epochs per rollout
        return hp["n_epochs"] * (b // hp["n_steps"])
    return None


rows = []
revs, devices, pkgs, pyvers = {}, {}, {}, {}
first_eps = {}
step0 = {}
eval_seed_sets = {}
for key in sorted(runs):
    env, algo, seed = key
    d = runs[key]
    cfg = read_json(os.path.join(d, "config.resolved.json"))
    ident = read_json(os.path.join(d, "identity.json"))
    meta = read_json(os.path.join(d, "metadata.json"))
    st = read_json(os.path.join(d, "status.json"))
    fs = read_json(os.path.join(d, "final_summary.json"))
    budget = cfg["budget"]["env_steps"]
    rec = {"env": env, "algo": algo, "seed": seed, "run_dir": os.path.relpath(d, REPO).replace("\\", "/")}

    h = plan.config_hash(cfg)
    rec["hash_ok"] = h == ident["config_hash"] == plan_hash[key] == fs["config_hash"]
    rec["config_equals_plan"] = json.dumps(plan.hashed_part(cfg), sort_keys=True) == json.dumps(
        plan.hashed_part(plan_cfg[f"{env}/{algo}/seed{seed}"]), sort_keys=True)
    if not rec["hash_ok"] or not rec["config_equals_plan"]:
        problem(f"{key}: config hash/config differs from plan")

    # seeds
    rec["seed_ok"] = (cfg["seed"]["train_seed"] == seed == ident["seed"] and seed in TRAIN_SEED_RANGES["baseline"]
                      and cfg["seed"]["train_env_reset_seed"] == train_env_seed(seed))
    if not rec["seed_ok"]:
        problem(f"{key}: seed fields inconsistent")

    # budget and step accounting
    ep = pd.read_csv(os.path.join(d, "train_episodes.csv"))
    pts = pd.read_csv(os.path.join(d, "eval_points.csv"))
    per = pd.read_csv(os.path.join(d, "eval_periodic.csv"))
    fin = pd.read_csv(os.path.join(d, "eval_final.csv"))
    upd = pd.read_csv(os.path.join(d, "train_updates.csv"))
    max_len = cfg["env"]["kwargs"]["max_steps"]
    last_ep_step = int(ep["global_step"].iloc[-1])
    rec.update(
        budget=budget, env_steps_done=st["env_steps_done"], summary_step=fs["global_step"],
        last_eval_step=int(pts["global_step"].iloc[-1]), n_eval_points=len(pts),
        eval_steps_ok=list(pts["global_step"]) == [i * cfg["eval"]["every_steps"] for i in range(cfg["eval"]["n_points"] + 1)],
        n_final=len(fin), n_periodic_rows=len(per),
        train_episodes=len(ep), train_episodes_summary=fs["train_episodes"],
        sum_ep_len=int(ep["length"].sum()), last_ep_step=last_ep_step,
        ep_steps_consistent=bool(int(ep["length"].sum()) == last_ep_step and ep["global_step"].is_monotonic_increasing
                                 and (np.diff(np.r_[0, ep["global_step"].values]) == ep["length"].values).all()),
        tail_steps_unfinished=budget - last_ep_step, max_episode_len_obs=int(ep["length"].max()),
        n_updates=st["n_updates"], n_updates_expected=expected_updates(cfg),
        last_logged_updates=int(upd["n_updates"].iloc[-1]),
    )
    ok_budget = (st["env_steps_done"] == budget == fs["global_step"] == rec["last_eval_step"] and rec["eval_steps_ok"]
                 and len(fin) == cfg["eval"]["final_episodes"] and len(pts) == cfg["eval"]["n_points"] + 1
                 and len(per) == len(pts) * cfg["eval"]["periodic_episodes"]
                 and rec["ep_steps_consistent"] and 0 <= rec["tail_steps_unfinished"] < max_len
                 and rec["train_episodes"] == rec["train_episodes_summary"])
    rec["budget_ok"] = bool(ok_budget)
    if not ok_budget:
        problem(f"{key}: budget/step accounting problem {rec}")
    rec["updates_ok"] = rec["n_updates"] == rec["n_updates_expected"]
    if not rec["updates_ok"]:
        problem(f"{key}: n_updates {rec['n_updates']} != expected {rec['n_updates_expected']}")

    # evaluation episode seeds
    per_seeds = sorted(set(per["episode_seed"]))
    fin_seeds = list(fin["episode_seed"])
    rec["eval_seeds_ok"] = (per_seeds == [EVAL_EPISODE_SEED_BASE["periodic"] + i for i in range(cfg["eval"]["periodic_episodes"])]
                            and fin_seeds == [EVAL_EPISODE_SEED_BASE["final"] + i for i in range(cfg["eval"]["final_episodes"])])
    if not rec["eval_seeds_ok"]:
        problem(f"{key}: evaluation episode seeds differ from the plan")

    # resumes, failures
    rec["state"] = st["state"]
    rec["verified"] = st.get("verified")
    rec["resumes"] = len(st.get("resumes", []))
    rec["restored_steps"] = st.get("restored_steps")
    rec["resume_metadata_files"] = len(glob.glob(os.path.join(d, "metadata_resume_*.json")))
    rec["pre_resume_bak_files"] = len(glob.glob(os.path.join(d, "*.pre-resume-*")))
    if rec["resumes"] or rec["restored_steps"] or rec["resume_metadata_files"] or rec["pre_resume_bak_files"]:
        problem(f"{key}: resumed run")

    # code revision, machine, device
    g = meta["git"]
    rec["git_rev"], rec["git_dirty"] = g["rev"], g["dirty"]
    revs.setdefault(g["rev"], []).append(key)
    dev = meta["device"]
    rec["device"] = dev["selected"]
    rec["gpu_name"] = dev.get("gpu_name")
    rec["is_gpu_run"] = dev["is_gpu_run"]
    want_gpu = env == "colour"
    rec["device_ok"] = bool(dev["is_gpu_run"] == want_gpu and (dev["selected"] == "cpu") == (not want_gpu))
    if not rec["device_ok"]:
        problem(f"{key}: device {dev}")
    rec["machine_role"] = ident["machine_role"]
    rec["hostname"] = meta["hardware"]["hostname"]
    rec["gpu_model"] = [x["name"] for x in meta["hardware"].get("gpus", [])]
    rec["cpu_logical"] = meta["hardware"]["cpu_count_logical"]
    rec["threads"] = meta["threads"]["torch_num_threads"]
    pkgs.setdefault(json.dumps(meta["packages"], sort_keys=True), []).append(key)
    pyvers.setdefault(meta["python"]["version"], []).append(key)
    rec["wandb_mode"] = st.get("wandb", {}).get("mode")
    rec["wall_clock_h"] = (pd.Timestamp(st["finished_utc"]) - pd.Timestamp(st["started_utc"])).total_seconds() / 3600

    # independence material
    first_eps[key] = ep[["length", "return_raw"]].head(50).to_numpy()
    step0[key] = per[per["eval_index"] == 0][["return_raw", "length"]].to_numpy()
    eval_seed_sets.setdefault(env, set()).add((tuple(per_seeds), tuple(fin_seeds)))
    rows.append(rec)

df = pd.DataFrame(rows)

# ------------------------------------------------------------------ 3. code revision
rev_list = sorted(revs)
code_diff = {}
for rev in rev_list:
    out = git("diff", "--stat", VALIDATED_REV, rev, "--", "*.py", "*.sh", "requirements.txt", "setup.py",
              "run_scripts/baseline/baseline_plan.json", ":(exclude)play_human").strip()
    all_files = git("diff", "--name-only", VALIDATED_REV, rev).split()
    code_diff[rev] = {"runs": len(revs[rev]), "code_diff_vs_validated": out or "(none)",
                      "all_changed_files_vs_validated": all_files,
                      "is_descendant_of_validated": subprocess.run(["git", "merge-base", "--is-ancestor", VALIDATED_REV, rev],
                                                                   cwd=REPO).returncode == 0}
    if out:
        problem(f"code at {rev} differs from validated {VALIDATED_REV}: {out}")
if len(rev_list) != 1:
    problem(f"runs use {len(rev_list)} code revisions: {rev_list}")
head_code_diff = git("diff", "--stat", rev_list[0], "HEAD", "--", "*.py", "*.sh", "requirements.txt", "setup.py",
                     "run_scripts/baseline/baseline_plan.json", ":(exclude)play_human").strip() or "(none)"

# ------------------------------------------------------------------ 4. seed independence
indep = {}
for env, algos in CELLS.items():
    for algo in algos:
        seqs = [first_eps[(env, algo, s)] for s in SEEDS]
        same_pairs = []
        for i in range(5):
            for j in range(i + 1, 5):
                n = min(len(seqs[i]), len(seqs[j]))
                if np.array_equal(seqs[i][:n], seqs[j][:n]):
                    same_pairs.append((SEEDS[i], SEEDS[j]))
        s0 = [step0[(env, algo, s)] for s in SEEDS]
        same_s0 = [(SEEDS[i], SEEDS[j]) for i in range(5) for j in range(i + 1, 5) if np.array_equal(s0[i], s0[j])]
        indep[f"{env}/{algo}"] = {"identical_first50_train_episode_pairs": same_pairs,
                                  "identical_step0_eval_pairs": same_s0}
        if same_pairs:
            problem(f"{env}/{algo}: seeds {same_pairs} produced identical first training episodes")
# same seed, different algorithms share the training-env reset seed (paired design): first episode should match
paired = {}
for env, algos in CELLS.items():
    for s in SEEDS:
        firsts = {a: tuple(first_eps[(env, a, s)][0]) for a in algos}
        paired[f"{env}/seed{s}"] = firsts
eval_seeds_identical = {env: len(v) == 1 for env, v in eval_seed_sets.items()}

# ------------------------------------------------------------------ 5. batch logs
batches = {}
for bdir in sorted(glob.glob(os.path.join(FULL, "_batches", "*"))):
    bs = pd.read_csv(os.path.join(bdir, "batch_status.tsv"), sep="\t")
    errs = {}
    for f in glob.glob(os.path.join(bdir, "logs", "*.err")):
        txt = open(f, encoding="utf-8", errors="replace").read()
        flagged = [ln for ln in txt.splitlines() if any(w in ln for w in ("Traceback", "Error", "error", "nan", "NaN"))]
        if flagged:
            errs[os.path.basename(f)] = flagged[:5]
    batches[os.path.basename(bdir)] = {"runs": len(bs), "nonzero_exit": int((bs["exit_code"] != 0).sum()),
                                       "stderr_flagged_lines": errs}
    if (bs["exit_code"] != 0).any():
        problem(f"batch {bdir} has nonzero exits")
machine_logs = {}
for f in glob.glob(os.path.join(FULL, "_machine_logs", "*", "*.log")):
    lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
    machine_logs[os.path.basename(f)] = {"last_line": lines[-1] if lines else "",
                                         "failed_lines": [ln for ln in lines if "FAILED" in ln]}

summary = {
    "manifest": manifest,
    "plan_file_check": plan_check,
    "n_runs": len(df),
    "all": {k: bool(df[k].all()) for k in ("hash_ok", "config_equals_plan", "seed_ok", "budget_ok", "updates_ok",
                                              "eval_seeds_ok", "device_ok")},
    "states": df["state"].value_counts().to_dict(),
    "resumed_runs": int((df["resumes"] > 0).sum()),
    "git_revisions": code_diff,
    "code_diff_rev_to_HEAD": head_code_diff,
    "git_dirty_runs": int(df["git_dirty"].sum()),
    "python_versions": {k: len(v) for k, v in pyvers.items()},
    "package_sets": len(pkgs),
    "packages": json.loads(next(iter(pkgs))) if len(pkgs) == 1 else list(pkgs),
    "machines": df.groupby("machine_role").agg(host=("hostname", "first"), gpu=("gpu_model", "first"),
                                               cpus=("cpu_logical", "first"), threads=("threads", "first"),
                                               runs=("algo", "size")).reset_index().to_dict("records"),
    "colour_gpu_names": sorted(set(df.loc[df.env == "colour", "gpu_name"].dropna())),
    "eval_episode_seeds_identical_across_runs": eval_seeds_identical,
    "seed_independence": indep,
    "paired_first_training_episode_by_seed": {k: {a: list(v) for a, v in d.items()} for k, d in paired.items()},
    "batches": batches,
    "machine_logs": machine_logs,
    "per_run": df.to_dict("records"),
    "problems": problems,
}
write_json(os.path.join(OUT, "reconcile.json"), summary)
cols = ["env", "algo", "seed", "budget", "env_steps_done", "train_episodes", "tail_steps_unfinished", "n_updates",
        "n_updates_expected", "device", "wall_clock_h", "git_rev"]
df[cols].to_csv(os.path.join(OUT, "reconcile_runs.csv"), index=False)
print(df[cols].to_string(index=False))
print("\nmanifest:", manifest_stdout_tail)
print("all checks:", summary["all"])
print("revisions:", {k: v["runs"] for k, v in code_diff.items()}, "code diff vs validated:",
      {k: v["code_diff_vs_validated"] for k, v in code_diff.items()})
print("problems:", len(problems))
