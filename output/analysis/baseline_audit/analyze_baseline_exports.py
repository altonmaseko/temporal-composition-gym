"""Audit the supplied baseline CSVs without changing source observations."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "wandb_phase_3_baseline_exports"
INITIAL = "596256061ffae5ea61dee31f2f5e70efc3bf67c7"
FIXED = "1760a6c5c2a86997c864e39810c79ca4db79f005"


def numeric(value):
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite value: {value!r}")
    return result


def analyze():
    with (SOURCE / "all_runs_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 24
    assert len({row["run_id"] for row in summary}) == 24
    runs = []
    hashes = {}
    for path in sorted(SOURCE.glob("*.csv")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for summary_line, row in enumerate(summary, 2):
        name = row["name"]
        algorithm = name.split("__")[1]
        env = row.get("env") or row.get("env_id")
        seed = int(float(row["seed"]))
        path = SOURCE / f"{name}_history.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            history = list(reader)

        def observations(key):
            return [
                {"csv_line": line, "step": numeric(record.get("global_step")), "value": numeric(record[key])}
                for line, record in enumerate(history, 2)
                if record.get(key) not in (None, "")
            ]

        custom = algorithm in ("drqn", "qrm")
        metric = "charts/episodic_return" if custom else "rollout/ep_rew_mean"
        returns = observations(metric)
        steps = [numeric(record["global_step"]) for record in history if record.get("global_step")]
        log_steps = [numeric(record["_step"]) for record in history if record.get("_step")]
        assert all(b >= a for a, b in zip(log_steps, log_steps[1:])), name
        requested = int(numeric(row.get("timesteps") or row.get("total_timesteps")))
        visual_drqn = algorithm == "drqn" and "SequentialColour" in env
        maintenance_rm = algorithm == "ppo_rm" and "DeferredMaintenance" in env
        fixed = visual_drqn or maintenance_rm
        loss_columns = [key for key in columns if key.startswith("losses/") or key == "train/loss"]
        loss_points = sum(len(observations(key)) for key in loss_columns)

        if not history:
            classification = "empty_history_checkpoint_skip_consistent"
        elif visual_drqn:
            classification = "no_observed_optimizer_updates_below_replay_warmup"
        elif algorithm in ("drqn", "qrm") and "HazardousDelivery" in env and seed > 1:
            classification = "warm_start_from_shared_checkpoint_not_independent"
        elif not returns:
            classification = "optimization_recorded_no_completed_episode_return"
        elif "DeferredMaintenance" in env:
            classification = "trained_but_sparse_or_stale_completed_episode_metrics"
        else:
            classification = "individual_training_trace_available"

        run = {
            "run_id": row["run_id"], "name": name, "environment": env,
            "algorithm": algorithm, "seed_label": seed,
            "commit": FIXED if fixed else INITIAL,
            "commit_attribution": "user-supplied historical W&B verification; git objects independently present",
            "summary_csv_line": summary_line,
            "history_file": path.name, "history_rows": len(history),
            "summary_history_step": numeric(row.get("_step")),
            "history_export_has_all_wandb_steps": (
                log_steps == list(range(int(log_steps[-1]) + 1))
                and numeric(row.get("_step")) == log_steps[-1]
            ) if log_steps else None,
            "configured_timesteps": requested,
            "original_intended_budget": 3_000_000 if "SequentialColour" in env else (2_000_000 if "DeferredMaintenance" in env else 1_000_000),
            "runtime_seconds": numeric(row.get("_runtime")),
            "first_logged_environment_step": min(steps) if steps else None,
            "last_logged_environment_step": max(steps) if steps else None,
            "summary_environment_step": numeric(row.get("global_step")),
            "return_metric": metric,
            "metric_kind": "individual_completed_training_episode" if custom else "moving_mean_of_up_to_100_completed_training_episodes",
            "return_points": len(returns),
            "minimum_observed_return_metric": min((p["value"] for p in returns), default=None),
            "maximum_observed_return_metric": max((p["value"] for p in returns), default=None),
            "final_summary_return_metric": numeric(row.get(metric)),
            "last_exported_return_point": returns[-1] if returns else None,
            "unique_return_metric_values": len({p["value"] for p in returns}),
            "optimization_metric_points": loss_points,
            "loss_columns": loss_columns,
            "observed_hd_returns_over_30": sum(p["value"] > 30 for p in returns) if custom and "HazardousDelivery" in env else None,
            "classification": classification,
            "history_columns": columns,
        }
        if visual_drqn:
            lengths = observations("charts/episodic_length")
            run["completed_episodes_recorded"] = len(returns)
            run["minimum_completed_episodes_before_updates"] = int(numeric(row["batch_size"])) + 1
            run["sum_recorded_episode_lengths"] = sum(point["value"] for point in lengths)
            run["completed_lengths_match_logged_endpoint"] = run["sum_recorded_episode_lengths"] == max(steps) + 1
            assert run["history_export_has_all_wandb_steps"]
            assert len(returns) < run["minimum_completed_episodes_before_updates"]
            assert loss_points == 0
            assert run["completed_lengths_match_logged_endpoint"]
        runs.append(run)

    groups = {}
    for run in runs:
        key = f"{run['environment']} | {run['algorithm']}"
        groups.setdefault(key, []).append(run)
    assert len(groups) == 8
    assert all(sorted(r["seed_label"] for r in group) == [1, 2, 3] for group in groups.values())
    result = {
        "audit_date": "2026-10-25",
        "scope": "Baseline runs only; no hyperparameter tuning or stress-testing results",
        "source_hashes_sha256": hashes,
        "run_count": len(runs),
        "initial_commit_records": sum(r["commit"] == INITIAL for r in runs),
        "fixed_commit_records": sum(r["commit"] == FIXED for r in runs),
        "empty_history_runs": sum(r["history_rows"] == 0 for r in runs),
        "nonempty_history_runs": sum(r["history_rows"] > 0 for r in runs),
        "runs_with_return_metrics": sum(r["return_points"] > 0 for r in runs),
        "warm_start_hd_runs": sum(r["classification"].startswith("warm_start") for r in runs),
        "runs": sorted(runs, key=lambda r: (r["environment"], r["algorithm"], r["seed_label"])),
    }
    (OUT / "baseline_export_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = analyze()
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "source_hashes_sha256")}, indent=2))
    for run in result["runs"]:
        print(run["environment"].split("/")[-1], run["algorithm"], run["seed_label"],
              run["history_rows"], run["return_points"], run["classification"])
