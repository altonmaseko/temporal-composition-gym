"""Local, offline end-to-end validation of the protocol-v2 baseline pipeline (Prompt 4).

Everything here is short, labelled smoke work on this machine: W&B runs offline (records are
decoded and checked, which is NOT proof of online upload), devices are CPU unless --device is
given, and all run directories live in a temporary work directory that is deleted afterwards.

Sections
  matrix      one smoke run per environment/algorithm combination (11): gradient updates,
              evaluations, restart checkpoint saved AND reloaded, finite metrics, budget
              accounting, artifact completeness, offline W&B records, smoke-vs-full differences
  resume      injected failure + resume for DQN, PPO, Recurrent PPO, DRQN and QRM; update counts
              must equal the uninterrupted run's
  collision   second seed and a second configuration never reuse or inherit the first run
  failure     deliberate failure and missing-metric cases are reported, never as success/zero
  config      invalid configurations are rejected
  envs        observations, rewards and events are finite and well-formed
  memory      image replay allocations projected for the full runs, measured smoke RSS
  manifest    the smoke matrix is complete in check_manifest
  pytest      (optional) the full semantic + pipeline test suite

    python run_scripts/baseline/validate_pipeline.py [--pytest] [--work-dir DIR] [--keep]
Exit code 0 only if every check passed.
"""
import argparse
import csv
import glob
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "run_scripts", "baseline")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402

from agents.common import baseline_plan as plan  # noqa: E402
from agents.common.run_io import git_info, read_json  # noqa: E402

CHECKS = []
ENV = dict(os.environ)


def check(section, name, ok, detail="", **data):
    CHECKS.append({"section": section, "check": name, "ok": bool(ok), "detail": detail, **data})
    print(f"[{'PASS' if ok else 'FAIL'}] {section}: {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def rows(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_entry(env_key, algo, seed, root, device, wandb_mode="offline", extra=()):
    script = os.path.join(REPO_ROOT, plan.ALGORITHMS[algo]["script"])
    cmd = [sys.executable, script, "--env", env_key, "--seed", str(seed), "--mode", "smoke", "--device", device,
           "--wandb-mode", wandb_mode, "--results-root", root, *extra]
    if algo == "dqn_ts":
        cmd.append("--task-state")
    before = set(glob.glob(os.path.join(root, "v2", "smoke", env_key, algo, "seed*__*")))
    t = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=ENV, cwd=tempfile.gettempdir())
    new = sorted(set(glob.glob(os.path.join(root, "v2", "smoke", env_key, algo, "seed*__*"))) - before)
    return res, (new[0] if len(new) == 1 else None), time.time() - t


# --------------------------------------------------------------------------- W&B offline records
def read_wandb_records(path):
    """Decode a .wandb file (LevelDB-style log of protobuf Records)."""
    from wandb.proto import wandb_internal_pb2 as pb
    data = open(path, "rb").read()
    if data[:4] != b":W&B":
        raise ValueError("not a W&B record file")
    pos, block, buf, out = 7, 32768, b"", []
    while pos + 7 <= len(data):
        left = block - (pos % block)
        if left < 7:
            pos += left
            continue
        _, length, typ = struct.unpack("<IHB", data[pos:pos + 7])
        pos += 7
        chunk = data[pos:pos + length]
        pos += length
        if typ == 0 and length == 0:
            continue
        buf += chunk
        if typ in (1, 4):
            r = pb.Record()
            r.ParseFromString(buf)
            out.append(r)
            buf = b""
    return out


def check_offline_wandb(section, rd, identity, final_summary, n_points):
    files = glob.glob(os.path.join(rd, "wandb", "offline-run-*", "run-*.wandb"))
    if len(files) != 1:
        return check(section, "offline W&B record file", False, f"found {len(files)}")
    recs = read_wandb_records(files[0])
    kind = lambda r: r.WhichOneof("record_type")  # noqa: E731
    run = next(r.run for r in recs if kind(r) == "run")
    key = lambda i: i.key or "/".join(i.nested_key)  # noqa: E731
    hist = [{key(i): i.value_json for i in r.history.item} for r in recs if kind(r) == "history"]
    evals = [h for h in hist if "eval/success_rate" in h]
    summary = {}
    for r in recs:
        if kind(r) == "summary":
            for u in r.summary.update:
                summary[key(u)] = u.value_json
    exit_codes = [r.exit.exit_code for r in recs if kind(r) == "exit"]
    problems = []
    if run.run_id != identity["wandb_id"]:
        problems.append("run id differs from identity.json")
    if run.project != "new_experiments" or not run.run_group.endswith("-smoke"):
        problems.append(f"project/group {run.project}/{run.run_group}")
    if len(evals) != n_points:
        problems.append(f"{len(evals)} eval history rows, expected {n_points}")
    if any("global_step" not in h for h in hist):
        problems.append("history row without global_step")
    fs = final_summary["final_eval"]["success_rate"]
    if "final_eval/success_rate" not in summary or not math.isclose(float(summary["final_eval/success_rate"]), fs):
        problems.append("final_eval/success_rate missing or different from final_summary.json")
    if exit_codes != [0]:
        problems.append(f"exit records {exit_codes}")
    return check(section, "offline W&B records (decoded; not proof of upload)", not problems,
                 "; ".join(problems) or f"{len(hist)} history rows, {len(evals)} eval rows, summary matches local files")


# --------------------------------------------------------------------------- checkpoint reload
def check_checkpoint_reload(section, rd, cfg):
    import torch
    snaps = sorted(glob.glob(os.path.join(rd, "checkpoints", "policy_step*")))
    if not snaps:
        return check(section, "checkpoint reload", False, "no policy snapshots")
    last = snaps[-1]
    restart = os.path.join(rd, "checkpoints", "restart")
    if cfg["algo"]["impl"] == "sb3":
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import DQN, PPO
        cls = {"dqn": DQN, "dqn_ts": DQN, "ppo": PPO, "ppo_rm": PPO, "recurrent_ppo": RecurrentPPO}[cfg["algo"]["name"]]
        a = cls.load(os.path.join(restart, "model.zip"), device="cpu").policy.state_dict()
        b = cls.load(last, device="cpu").policy.state_dict()
        extra = "replay.pkl" in os.listdir(restart) if cls is DQN else True
    else:
        a = torch.load(os.path.join(restart, "state.pt"), weights_only=False, map_location="cpu")["q"]
        b = torch.load(last, weights_only=False, map_location="cpu")
        extra = "replay.npz" in os.listdir(restart)
    same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    finite = all(torch.isfinite(v).all() for v in a.values() if v.is_floating_point())
    step = read_json(os.path.join(restart, "state.json"))["global_step"]
    return check(section, "checkpoint saved and reloaded", same and finite and extra and step == cfg["budget"]["env_steps"],
                 f"restart step {step}; reloaded weights equal the final snapshot: {same}; finite: {finite}; "
                 f"replay saved: {extra}")


# --------------------------------------------------------------------------- sections
def smoke_differences(env_key, algo):
    full, smoke = plan.resolve_config(env_key, algo, 1), plan.resolve_config(env_key, algo, 1, "smoke")
    diffs = {}

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        elif a != b:
            diffs[path] = {"full": a, "smoke": b}
    walk(full, smoke, "")
    diffs.pop("mode", None)
    return diffs


def section_matrix(root, device):
    results = {}
    for role in plan.ROLES:
        for env_key, algo in plan.ROLE_MAIN_RUNS[role] + plan.ROLE_CONTROL_RUNS[role]:
            sec = f"matrix {env_key}/{algo}"
            res, rd, secs = run_entry(env_key, algo, 1, root, device)
            if not check(sec, "run exits 0 with one new run directory", res.returncode == 0 and rd,
                         f"exit {res.returncode} in {secs:.0f}s" + ("" if res.returncode == 0 else
                                                                    f": {res.stderr.strip().splitlines()[-1:]}")):
                continue
            cfg = read_json(os.path.join(rd, "config.resolved.json"))
            st = read_json(os.path.join(rd, "status.json"))
            ident = read_json(os.path.join(rd, "identity.json"))
            final = read_json(os.path.join(rd, "final_summary.json"))
            ep, upd = rows(os.path.join(rd, "train_episodes.csv")), rows(os.path.join(rd, "train_updates.csv"))
            pts, fin = rows(os.path.join(rd, "eval_points.csv")), rows(os.path.join(rd, "eval_final.csv"))
            per = rows(os.path.join(rd, "eval_periodic.csv"))
            resrc = rows(os.path.join(rd, "resources.csv"))
            budget, n_pts = cfg["budget"]["env_steps"], cfg["eval"]["n_points"] + 1
            check(sec, "learned (gradient updates > 0)", st.get("n_updates", 0) > 0, f"{st.get('n_updates')} updates")
            check(sec, "artifacts complete", st["state"] == "complete" and st.get("verified") and len(pts) == n_pts
                  and len(fin) == cfg["eval"]["final_episodes"] and len(per) == n_pts * cfg["eval"]["periodic_episodes"],
                  f"{len(pts)} eval points, {len(per)} periodic + {len(fin)} final eval episodes")
            lengths = [int(r["length"]) for r in ep]
            max_len = cfg["env"]["kwargs"]["max_steps"]
            check(sec, "budget accounting", st["env_steps_done"] == budget and sum(lengths) <= budget
                  and sum(lengths) > budget - max_len and int(pts[-1]["global_step"]) == budget,
                  f"{st['env_steps_done']}/{budget} steps; {len(lengths)} finished training episodes cover "
                  f"{sum(lengths)} steps (rest = episode in progress); evaluation steps not counted")
            losses = [float(r["loss"]) for r in upd if r["loss"]]
            rets = [float(r["return_raw"]) for r in ep + per + fin]
            check(sec, "finite losses and returns", losses and all(map(math.isfinite, losses))
                  and all(map(math.isfinite, rets)), f"{len(losses)} loss rows, {len(rets)} returns")
            check_checkpoint_reload(sec, rd, cfg)
            check_offline_wandb(sec, rd, ident, final, n_pts)
            rss = max(float(r["rss_gb"]) for r in resrc if r["rss_gb"])
            results[f"{env_key}/{algo}"] = {"run_dir": rd, "seconds": round(secs, 1), "n_updates": st["n_updates"],
                                           "peak_rss_gb": round(rss, 3), "device": cfg and read_json(
                                               os.path.join(rd, "metadata.json"))["device"]["selected"],
                                           "smoke_vs_full": smoke_differences(env_key, algo)}
    return results


def section_resume(root, device, matrix):
    for env_key, algo in (("delivery", "dqn"), ("maintenance", "ppo"), ("maintenance", "recurrent_ppo"),
                          ("delivery", "drqn"), ("delivery", "qrm")):
        sec = f"resume {env_key}/{algo}"
        ref = matrix.get(f"{env_key}/{algo}")
        res, rd, _ = run_entry(env_key, algo, 1, root, device, "disabled", ("--debug-fail-at-step", "3000"))
        st = read_json(os.path.join(rd, "status.json")) if rd else {}
        check(sec, "injected failure recorded", res.returncode == 1 and st.get("state") == "failed",
              st.get("error", ""))
        restart = read_json(os.path.join(rd, "checkpoints", "restart", "state.json"))["global_step"]
        res2, _, _ = run_entry(env_key, algo, 1, os.path.join(root, "_unused"), device, "disabled", ("--resume", rd))
        st = read_json(os.path.join(rd, "status.json"))
        steps = [int(r["global_step"]) for r in rows(os.path.join(rd, "eval_points.csv"))]
        ok = (res2.returncode == 0 and st["state"] == "complete" and st["restored_steps"] == restart
              and st["restored_steps"] + st["new_steps_this_session"] == st["budget"]
              and steps == sorted(set(steps)) and ref is not None and st["n_updates"] == ref["n_updates"])
        check(sec, "resumed run completes with the uninterrupted run's update count", ok,
              f"restored {st.get('restored_steps')} + new {st.get('new_steps_this_session')} = {st.get('budget')}; "
              f"updates {st.get('n_updates')} vs uninterrupted {ref and ref['n_updates']}; eval steps {steps}")
        step0 = rows(os.path.join(rd, "eval_points.csv"))[0]
        ref0 = rows(os.path.join(ref["run_dir"], "eval_points.csv"))[0]
        check(sec, "same seed+config -> same step-0 evaluation", step0 == ref0)


def section_collision(root, device, matrix):
    sec = "collision"
    ref = matrix["delivery/dqn"]
    r2, d2, _ = run_entry("delivery", "dqn", 2, root, device, "disabled")
    r3, d3, _ = run_entry("delivery", "dqn", 1, root, device, "disabled", ("--set", "algo.hyperparams.learning_rate=0.001"))
    dirs = [ref["run_dir"], d2, d3]
    ids = [read_json(os.path.join(d, "identity.json")) for d in dirs if d]
    check(sec, "two seeds and two configs get separate run directories and identities",
          r2.returncode == 0 and r3.returncode == 0 and len(set(dirs)) == 3 and len({i["run_id"] for i in ids}) == 3
          and len({i["config_hash"] for i in ids}) == 3 and len({i["wandb_id"] for i in ids}) == 3)
    s3 = read_json(os.path.join(d3, "status.json"))
    same0 = rows(os.path.join(d3, "eval_points.csv"))[0] == rows(os.path.join(ref["run_dir"], "eval_points.csv"))[0]
    check(sec, "second config starts fresh (nothing inherited)", s3["restored_steps"] == 0 and same0
          and s3["n_updates"] == ref["n_updates"],
          "step-0 evaluation equals the first run's untrained policy; full update count from scratch")
    files = set(os.listdir(d3)) | set(os.listdir(os.path.join(d3, "checkpoints")))
    check(sec, "no shared model directory", not os.path.exists(os.path.join(tempfile.gettempdir(), "models"))
          and "restart" in files)


def section_failure(root, device, matrix):
    import check_manifest
    from agents.common.tracking import summarise
    sec = "failure"
    res, rd, _ = run_entry("delivery", "dqn", 3, root, device, "disabled", ("--debug-fail-at-step", "500"))
    st = read_json(os.path.join(rd, "status.json"))
    check(sec, "deliberate failure -> state failed, no summary, nonzero exit",
          res.returncode == 1 and st["state"] == "failed" and not os.path.exists(os.path.join(rd, "final_summary.json")))
    m = check_manifest.build_manifest([root], "smoke", [3])
    e = next(x for x in m["entries"] if x["env"] == "delivery" and x["algo"] == "dqn")
    check(sec, "manifest reports the failed run as failed (not complete, not zero)", e["state"] == "failed",
          "; ".join(sorted({p for a in e["attempts"] for p in a["problems"]})))

    copy_root = os.path.join(root, "_missing_metric")
    src = matrix["delivery/dqn"]["run_dir"]
    rel = os.path.relpath(src, root)
    for name, mutate in (("eval_final.csv removed", lambda d: os.remove(os.path.join(d, "eval_final.csv"))),
                         ("last evaluation point removed", lambda d: _drop_last_row(os.path.join(d, "eval_points.csv"))),
                         ("final_summary.json removed", lambda d: os.remove(os.path.join(d, "final_summary.json")))):
        shutil.rmtree(copy_root, ignore_errors=True)
        dst = os.path.join(copy_root, rel)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoints", "wandb"))
        mutate(dst)
        ok, problems, _ = check_manifest.verify_run(dst)
        m = check_manifest.build_manifest([copy_root], "smoke", [1])
        e = next(x for x in m["entries"] if x["env"] == "delivery" and x["algo"] == "dqn")
        check(sec, f"missing metric ({name}) -> not verified, not counted", not ok and e["state"] == "failed",
              "; ".join(problems))
    try:
        summarise([])
        check(sec, "an evaluation with no episodes cannot produce a zero summary", False)
    except (ValueError, ZeroDivisionError, IndexError) as exc:
        check(sec, "an evaluation with no episodes cannot produce a zero summary", True, type(exc).__name__)
    shutil.rmtree(copy_root, ignore_errors=True)


def _drop_last_row(path):
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    open(path, "w", encoding="utf-8").write("\n".join(lines[:-1]) + "\n")


def section_config():
    sec = "config"
    bad = [
        ("unknown override", lambda: plan.resolve_config("delivery", "dqn", 1, overrides={"algo.hyperparams.nope": 1})),
        ("override of the seed", lambda: plan.resolve_config("delivery", "dqn", 1, overrides={"seed.train_seed": 2})),
        ("pilot W&B project", lambda: plan.resolve_config("delivery", "dqn", 1, wandb_project="temporal-composition-gym")),
        ("tuning-range seed", lambda: plan.resolve_config("delivery", "dqn", 150)),
        ("SB3 budget rounding", lambda: plan.resolve_config("maintenance", "ppo", 1, overrides={"budget.env_steps": 2_000_000})),
        ("algorithm outside the plan", lambda: plan.resolve_config("maintenance", "dqn", 1)),
        ("env readiness range impossible", lambda: _make_env_bad()),
    ]
    for name, fn in bad:
        try:
            fn()
            check(sec, f"rejects {name}", False)
        except (KeyError, ValueError) as exc:
            check(sec, f"rejects {name}", True, f"{type(exc).__name__}: {str(exc)[:90]}")
    check(sec, "baseline_plan.json matches the code", plan.main(["write", "--check"]) == 0)


def _make_env_bad():
    from agents.common.env_factory import make_env
    make_env("maintenance", overrides={"readiness_delay_range": (1, 4)})


def section_envs():
    from agents.common.env_factory import make_env
    for key in ("delivery", "colour", "maintenance"):
        env = make_env(key)
        rng = np.random.default_rng(0)
        env.action_space.seed(0)
        bad, n, events = [], 0, 0
        for ep in range(3):
            obs, info = env.reset(seed=3_000_000 + ep)
            while True:
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                n += 1
                if not np.all(np.isfinite(np.asarray(obs, dtype=np.float64))) or not env.observation_space.contains(obs):
                    bad.append("observation")
                if not math.isfinite(r):
                    bad.append("reward")
                if not all(isinstance(e, tuple) and len(e) == 2 and isinstance(e[0], str) for e in info["events"]):
                    bad.append("event format")
                events += len(info["events"])
                if term or trunc:
                    if info.get("termination_cause") is None or "is_success" not in info:
                        bad.append("termination info")
                    break
        check("envs", f"{key}: finite, in-space observations and well-formed events", not bad,
              f"{n} random steps, {events} events" + (f"; problems: {sorted(set(bad))}" if bad else ""))


def section_memory(matrix):
    import gymnasium as gym
    from stable_baselines3.common.buffers import ReplayBuffer
    from agents.common.replay import FrameReplay
    sec = "memory"
    img = gym.spaces.Box(0, 255, (3, 84, 84), np.uint8)
    n = 1000
    sb3 = ReplayBuffer(n, img, gym.spaces.Discrete(4), device="cpu", handle_timeout_termination=True)
    sb3_bytes = sum(getattr(sb3, a).nbytes for a in ("observations", "next_observations", "actions", "rewards",
                                                     "dones", "timeouts")) / n
    drqn = plan.resolve_config("colour", "drqn", 1)["algo"]["hyperparams"]
    qrm = plan.resolve_config("colour", "qrm_cue", 1)["algo"]["hyperparams"]
    dq = plan.resolve_config("colour", "dqn", 1)["algo"]["hyperparams"]
    fr_d = FrameReplay(n, (84, 84, 3), np.uint8, hidden_shape=(2, drqn["lstm_dim"])).nbytes / n
    fr_q = FrameReplay(n, (84, 84, 3), np.uint8, extras={k: ((), np.int16) for k in ("head", "next_head", "cls", "w", "c_next")}).nbytes / n
    proj = {"dqn/dqn_ts colour (SB3, obs + next_obs)": sb3_bytes * dq["buffer_size"] / 1e9,
            "drqn colour (frames + LSTM state)": fr_d * drqn["replay_frames"] / 1e9,
            "qrm_cue colour (frames)": fr_q * qrm["replay_frames"] / 1e9}
    measured = {k: v["peak_rss_gb"] for k, v in matrix.items() if k.startswith("colour/")}
    check(sec, "projected full-run image replay allocation", all(v < 5.0 for v in proj.values()),
          ", ".join(f"{k}: {v:.2f} GB" for k, v in proj.items()), projected_gb=proj, smoke_peak_rss_gb=measured)


def section_manifest(root):
    import check_manifest
    m = check_manifest.build_manifest([root], "smoke", [1])
    check("manifest", "all 11 planned combinations complete (smoke, seed 1)", m["all_complete"], str(m["counts"]))


def section_pytest(work):
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                          f"--basetemp={os.path.join(work, '_pytest')}"], capture_output=True, text=True,
                         cwd=REPO_ROOT, env=ENV)
    tail = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-300:]
    check("pytest", "semantic + pipeline test suite", res.returncode == 0, tail)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None, help="scratch directory (default: a new temp dir, deleted afterwards)")
    p.add_argument("--keep", action="store_true", help="keep the scratch run directories")
    p.add_argument("--device", default="cpu", help="device for the smoke runs (local default cpu)")
    p.add_argument("--pytest", action="store_true", help="also run the full pytest suite")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "run_scripts", "baseline", "validation",
                                                 "pipeline_validation_v2.json"))
    args = p.parse_args()
    work = args.work_dir or tempfile.mkdtemp(prefix="tcg_validate_")
    os.makedirs(work, exist_ok=True)
    root = os.path.join(work, "results")
    t0 = time.time()
    g = git_info()
    print(f"validate_pipeline: rev {g['rev']} dirty={g['dirty']} work={work}", flush=True)
    matrix = {}
    for name, fn in (("matrix", lambda: matrix.update(section_matrix(root, args.device))),
                     ("resume", lambda: section_resume(root, args.device, matrix)),
                     ("collision", lambda: section_collision(root, args.device, matrix)),
                     ("failure", lambda: section_failure(root, args.device, matrix)),
                     ("config", section_config), ("envs", section_envs),
                     ("memory", lambda: section_memory(matrix)), ("manifest", lambda: section_manifest(root))):
        try:
            fn()
        except Exception as exc:                                         # noqa: BLE001
            import traceback
            check(name, "section ran", False, f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    if args.pytest:
        section_pytest(work)
    failed = [c for c in CHECKS if not c["ok"]]
    report = {
        "protocol_version": "v2", "kind": "local offline pipeline validation (smoke settings, not production)",
        "git": g, "python": sys.version.split()[0], "device": args.device, "minutes": round((time.time() - t0) / 60, 1),
        "passed": not failed, "n_checks": len(CHECKS), "n_failed": len(failed),
        "wandb_note": "W&B ran offline; records were decoded and checked locally. Online upload is not verified.",
        "smoke_runs": {k: {kk: vv for kk, vv in v.items() if kk != "run_dir"} for k, v in matrix.items()},
        "checks": CHECKS,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed in {report['minutes']} min; report: {args.out}")
    if not args.keep and args.work_dir is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
