"""Independent verification of stress runs, the expected-vs-found manifest, and safe pruning.

A run counts only if its own files prove it: plan and config hashes, identity (level, seed,
role, slot), exact budget and learner updates, every evaluation point and episode seed, the
stress per-episode fields, finite metrics and the final evaluations. Anything else is
unavailable, never zero.
"""
import csv
import math
import shutil
import statistics
from collections import Counter
from pathlib import Path

from agents.common import baseline_plan as base
from agents.common.run_io import STRESS_EPISODE_FIELDS, read_json
from agents.common.seeding import eval_episode_seeds
from agents.common.stress_io import run_parent, safe_root
from agents.common.stress_plan import run_config
from agents.common.tuning_records import expected_updates


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite metric")
    return value


def verify_run(path, cfg, plan_hash, entry):
    path = Path(path)
    result = {"run_dir": str(path), "verified": False, "problems": []}
    try:
        ident, actual, status = [read_json(str(path / n)) for n in ("identity.json", "config.resolved.json",
                                                                     "status.json")]
        result["state"] = status.get("state", "unknown")
        h = base.config_hash(cfg)
        if base.config_hash(actual) != h or ident["config_hash"] != h or ident.get("plan_hash") != plan_hash:
            raise ValueError("configuration/plan hash mismatch")
        want = {"phase": "stress", "mode": cfg["mode"], "env": cfg["env"]["key"], "algo": cfg["algo"]["name"],
                "seed": cfg["seed"]["train_seed"], "level": cfg["stress"]["level"], "role": entry["role"],
                "machine_role": entry["role"], "slot": entry["slot"], "round": entry["round"]}
        bad = [k for k, v in want.items() if ident.get(k) != v]
        if bad:
            raise ValueError(f"identity mismatch: {bad}")
        if status.get("state") != "complete" or status.get("verified") is not True:
            raise ValueError(f"run status {status.get('state')}: {status.get('error')}")
        summary = read_json(str(path / "final_summary.json"))
        budget, ev = cfg["budget"]["env_steps"], cfg["eval"]
        updates = expected_updates(cfg)
        if not (status.get("final_eval_done") is True and status["env_steps_done"] == summary["global_step"] == budget
                and status["n_updates"] == summary["n_updates"] == updates and updates > 0
                and summary["config_hash"] == h and summary["run_id"] == ident["run_id"]):
            raise ValueError("budget, update count or summary identity mismatch")
        steps = list(range(0, budget + 1, ev["every_steps"]))
        limit = cfg["env"]["kwargs"]["max_steps"]
        within = cfg["stress"]["within_steps"]
        for sfx in ("", "_noisy"):
            points = _rows(path / f"eval_points{sfx}.csv")
            if [int(r["global_step"]) for r in points] != steps or \
                    [int(r["eval_index"]) for r in points] != list(range(ev["n_points"] + 1)):
                raise ValueError("missing/duplicate/misplaced evaluation point")
            per = _rows(path / f"eval_periodic{sfx}.csv")
            seeds = eval_episode_seeds(ev["periodic_seed_block"], ev["periodic_episodes"])
            if [(int(r["eval_index"]), int(r["global_step"]), int(r["episode_seed"])) for r in per] != \
                    [(i, s, seed) for i, s in enumerate(steps) for seed in seeds]:
                raise ValueError("periodic episode count/seeds/cadence mismatch")
            for i, point in enumerate(points):
                rows = per[i * len(seeds):(i + 1) * len(seeds)]
                if int(point["n_episodes"]) != len(seeds):
                    raise ValueError("incorrect point episode count")
                for out, raw in (("success_rate", "success"), ("progress_mean", "progress"), ("return_mean", "return_raw")):
                    if not math.isclose(_num(point[out]), statistics.mean(_num(r[raw]) for r in rows),
                                        rel_tol=1e-10, abs_tol=1e-8):
                        raise ValueError("point summary differs from episode records")
            final = _rows(path / f"eval_final{sfx}.csv")
            if [int(r["episode_seed"]) for r in final] != eval_episode_seeds(ev["final_seed_block"], ev["final_episodes"]) \
                    or any(int(r["global_step"]) != budget for r in final):
                raise ValueError("final evaluation count/seeds/step mismatch")
            fs = summary["final_eval" + sfx]
            if fs["n_episodes"] != len(final):
                raise ValueError("final summary count mismatch")
            for out, raw in (("success_rate", "success"), ("progress_mean", "progress"), ("return_mean", "return_raw")):
                if not math.isclose(_num(fs[out]), statistics.mean(_num(r[raw]) for r in final),
                                    rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError("final summary differs from records")
            for row in per + final:
                if _num(row["success"]) not in (0, 1) or _num(row["progress"]) < 0 \
                        or not 0 < _num(row["length"]) <= limit:
                    raise ValueError("invalid episode metrics")
                if any(k not in row for k in STRESS_EPISODE_FIELDS) or _num(row["required"]) < 1 \
                        or _num(row["progress"]) > _num(row["required"]):
                    raise ValueError("missing or invalid required count")
                if within is not None:
                    if _num(row["progress_at_4000"]) > _num(row["progress"]) or \
                            _num(row["success_by_4000"]) not in (0, 1):
                        raise ValueError("invalid within-4,000-step metrics")
                    _num(row["return_at_4000"])
                elif any(row[k] for k in STRESS_EPISODE_FIELDS[1:]):
                    raise ValueError("within-limit metrics on a level whose limit was not raised")
        logs = _rows(path / "train_updates.csv")
        if not logs or int(logs[-1]["n_updates"]) != updates:
            raise ValueError("updates log incomplete")
        for row in logs:
            for key in ("loss", "q_mean", "grad_norm"):
                if row.get(key):
                    _num(row[key])
        if not any(r.get("loss") for r in logs):
            raise ValueError("no measured learner loss")
        policy = path / "checkpoints"
        result["final_policy"] = sorted(p.name for p in policy.glob(f"policy_step{budget:09d}.*")) if policy.exists() else []
        result.update(verified=True, state="complete", config_hash=h, n_updates=updates)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result["problems"].append(str(exc))
    return result


def inspect_run(root, cfg, plan_hash, entry):
    root = safe_root(root)
    attempts = []
    parent = run_parent(root, cfg)
    for path in sorted(parent.glob("a*")) if parent.exists() else []:
        if path.is_dir():
            rec = verify_run(path, cfg, plan_hash, entry)
            rec["run_dir"] = path.relative_to(root).as_posix()
            attempts.append(rec)
    good = [r for r in attempts if r["verified"]]
    if len(good) == 1:
        return {**good[0], "attempts": attempts}
    state = "duplicate" if good else "missing" if not attempts else "failed"
    if not good and any(r.get("state") in ("running", "interrupted") for r in attempts):
        state = "unfinished"
    return {"state": state, "attempts": attempts}


def manifest(root, plan, mode, role=None, level=None):
    root = safe_root(root)
    entries = []
    expected_parents = set()
    for entry in plan["runs"]:
        _, cfg = run_config(plan, entry["env"], entry["algo"], entry["level"], entry["seed"], mode)
        expected_parents.add(str(run_parent(root, cfg).resolve()))
        if (role and entry["role"] != role) or (level and entry["level"] != level):
            continue
        rec = inspect_run(root, cfg, plan["plan_hash"], entry)
        entries.append({k: entry[k] for k in ("role", "slot", "round", "env", "algo", "level", "seed")} | rec)
    base_dir = root / "v3" / mode
    unexpected = [str(p.relative_to(root)) for p in base_dir.glob("*/*/*/s*/a*/identity.json")
                  if str(p.parent.parent.resolve()) not in expected_parents] if base_dir.exists() else []
    return {"protocol_version": "v3", "phase": "stress", "mode": mode, "plan_hash": plan["plan_hash"],
            "filters": {"role": role, "level": level}, "expected": len(entries),
            "counts": dict(Counter(r["state"] for r in entries)),
            "by_role": {r: dict(Counter(e["state"] for e in entries if e["role"] == r)) for r in ("a", "b", "c")},
            "by_level": {lv: dict(Counter(e["state"] for e in entries if e["level"] == lv))
                         for lv in ("easy", "default", "hard1", "hard2")},
            "entries": entries, "unexpected_runs": unexpected,
            "all_complete": all(r["state"] == "complete" for r in entries) and not unexpected}


def prune_completed(root, cfg, plan_hash, entry):
    """Delete restart/replay data of an independently verified run; keep policies and records."""
    root = safe_root(root)
    record = inspect_run(root, cfg, plan_hash, entry)
    if record["state"] != "complete":
        return False
    attempt = (root / record["run_dir"]).resolve()
    if not attempt.is_relative_to(root) or attempt.parent != run_parent(root, cfg).resolve():
        raise ValueError("attempt outside the registered run")
    checkpoints = attempt / "checkpoints"
    if checkpoints.is_symlink():
        raise ValueError("unsafe checkpoints directory")
    for name in ("restart", "restart.old", "restart.tmp"):
        target = checkpoints / name
        if target.is_symlink() or target.resolve().parent != checkpoints.resolve():
            raise ValueError("unsafe restart path")
        if target.exists():
            shutil.rmtree(target)
    return True
