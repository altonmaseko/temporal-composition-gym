"""Read-only reconciliation of Optuna studies, CSV exports, and per-run histories."""

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "run_scripts/hyperparameter_tuning"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def read_study(algorithm):
    path = SOURCE / "results" / f"{algorithm}_tuning.db"
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    rows = db.execute(
        "SELECT t.trial_id,t.number,t.state,t.datetime_start,t.datetime_complete,v.value "
        "FROM trials t JOIN trial_values v ON t.trial_id=v.trial_id ORDER BY t.number"
    ).fetchall()
    trials = []
    for identifier, number, state, started, completed, value in rows:
        params = {}
        for name, stored, distribution in db.execute(
            "SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?", (identifier,)
        ):
            description = json.loads(distribution)
            if description["name"] == "CategoricalDistribution":
                params[name] = description["attributes"]["choices"][int(stored)]
            else:
                params[name] = stored
        trials.append({"trial_id": identifier, "number": number, "state": state,
                       "datetime_start_utc_assumption": started,
                       "datetime_complete_utc_assumption": completed,
                       "value": value, "params": params})
    metadata = {
        "study_name": db.execute("SELECT study_name FROM studies").fetchone()[0],
        "direction": db.execute("SELECT direction FROM study_directions").fetchone()[0],
        "version_info": db.execute("SELECT * FROM version_info").fetchall(),
        "intermediate_value_count": db.execute("SELECT count(*) FROM trial_intermediate_values").fetchone()[0],
        "user_attribute_count": db.execute("SELECT count(*) FROM trial_user_attributes").fetchone()[0],
    }
    db.close()
    return trials, metadata


def analyze():
    summary, _ = read_csv(SOURCE / "wandb_exports/all_runs_summary.csv")
    assert len(summary) == 60
    used_runs = set()
    output = {"date": "2026-10-25", "scope": "Hyperparameter tuning only",
              "time_alignment": "Naive Optuna datetimes treated as UTC, consistent with W&B created_at and user-provided +02:00 conversions",
              "studies": {}}
    for algorithm, expected in (("qrm", 15), ("drqn", 15), ("ppo_rm", 20)):
        trials, metadata = read_study(algorithm)
        csv_rows, _ = read_csv(SOURCE / "results" / f"{algorithm}_tuning_results.csv")
        assert len(trials) == len(csv_rows) == expected
        for trial, csv_row in zip(trials, csv_rows):
            assert trial["number"] == int(csv_row["number"])
            assert close(trial["value"], csv_row["value"])
            assert trial["state"] == csv_row["state"] == "COMPLETE"
            assert datetime.fromisoformat(trial["datetime_start_utc_assumption"]) == datetime.fromisoformat(csv_row["datetime_start"])
            assert datetime.fromisoformat(trial["datetime_complete_utc_assumption"]) == datetime.fromisoformat(csv_row["datetime_complete"])
            for key, value in trial["params"].items():
                assert close(value, csv_row[f"params_{key}"])
            start = datetime.fromisoformat(trial["datetime_start_utc_assumption"]).replace(tzinfo=timezone.utc)
            end = datetime.fromisoformat(trial["datetime_complete_utc_assumption"]).replace(tzinfo=timezone.utc)
            matches = []
            for row in summary:
                if f"__{algorithm}__{trial['number']}__" not in row["name"]:
                    continue
                created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                # Exported W&B creation times have second precision, while Optuna keeps microseconds.
                if not start - timedelta(seconds=1) <= created <= end:
                    continue
                if not all(row.get(key) and close(value, row[key]) for key, value in trial["params"].items()):
                    continue
                matches.append(row)
            assert len(matches) == 1, (algorithm, trial["number"], len(matches))
            run = matches[0]
            used_runs.add(run["run_id"])
            history_path = SOURCE / "wandb_exports" / f"{run['name']}_history.csv"
            history, columns = read_csv(history_path)

            def series(key):
                return [float(row[key]) for row in history if row.get(key)]

            log_indices = series("_step")
            full = bool(log_indices) and log_indices == list(range(int(log_indices[-1]) + 1)) and close(log_indices[-1], run["_step"])
            returns = series("charts/episodic_return")
            steps = series("global_step")
            losses = series("losses/td_loss" if algorithm == "drqn" else "losses/qrm_loss")
            trial["run"] = {
                "run_id": run["run_id"], "name": run["name"], "created_at": run["created_at"],
                "state": run["state"], "history_file": history_path.name,
                "history_rows": len(history), "complete_exported_wandb_steps": full,
                "return_samples": len(returns), "optimization_samples": len(losses),
                "first_exported_environment_step": min(steps) if steps else None,
                "last_exported_environment_step": max(steps) if steps else None,
                "summary_environment_step": float(run["global_step"]) if run.get("global_step") else None,
                "last_summary_episode_return": float(run["charts/episodic_return"]) if run.get("charts/episodic_return") else None,
                "configured_budget": float(run["total_timesteps"]) if run.get("total_timesteps") else None,
                "runtime_seconds": float(run["_runtime"]), "history_columns": columns,
                "last_ten_mean_exact_when_unsampled": sum(returns[-10:]) / len(returns[-10:]) if full and returns else None,
            }
            if algorithm == "drqn":
                trial["optimizer_updates_evidenced"] = bool(losses)
                trial["no_learning_replay_gate"] = bool(full and not losses and len(returns) <= trial["params"]["batch_size"])
                final_return = trial["run"]["last_summary_episode_return"]
                bound = (final_return + 9 * 10) / 10 if final_return is not None else None
                trial["upper_bound_own_terminal_last10_mean"] = bound
                trial["objective_exceeds_own_terminal_tail_bound"] = bool(bound is not None and trial["value"] > bound + 1e-9)
            if algorithm == "qrm":
                trial["optimizer_updates_evidenced"] = bool(losses)
                integral_returns = all(close(value, round(value)) for value in returns)
                trial["objective_not_on_required_tenth_grid"] = bool(len(returns) >= 10 and integral_returns and not close(trial["value"] * 10, round(trial["value"] * 10)))

        events = []
        for trial in trials:
            events.append((datetime.fromisoformat(trial["datetime_start_utc_assumption"]), 1))
            events.append((datetime.fromisoformat(trial["datetime_complete_utc_assumption"]), -1))
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        best = max(trials, key=lambda trial: trial["value"])
        output["studies"][algorithm] = {"metadata": metadata, "trial_count": expected,
            "observed_peak_overlapping_trials": peak,
            "best_trial_number": best["number"], "best_value": best["value"],
            "best_params": best["params"], "trials": trials}
    assert len(used_runs) == 50
    output["unmatched_wandb_records"] = [
        {"run_id": row["run_id"], "name": row["name"], "created_at": row["created_at"], "state": row["state"]}
        for row in summary if row["run_id"] not in used_runs
    ]
    sources = sorted((SOURCE / "results").glob("*.db")) + sorted((SOURCE / "results").glob("*.csv")) + sorted((SOURCE / "wandb_exports").glob("*.csv"))
    output["source_sha256"] = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    (OUT / "tuning_export_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    result = analyze()
    for algorithm, study in result["studies"].items():
        print(algorithm, study["trial_count"], "DB/CSV reconciled and W&B joined; best",study["best_trial_number"],study["best_value"], "peak overlap", study["observed_peak_overlapping_trials"])
        if algorithm != "ppo_rm":
            print("  trials with optimization metrics:", sum(t["optimizer_updates_evidenced"] for t in study["trials"]))
            print("  inconsistent own-tail objectives:", [t["number"] for t in study["trials"] if t.get("objective_exceeds_own_terminal_tail_bound") or t.get("objective_not_on_required_tenth_grid")])
        if algorithm == "drqn":
            print("  no-learning gate:", [t["number"] for t in study["trials"] if t["no_learning_replay_gate"]])
    print("Excluded unmatched preliminary/verification records:",len(result["unmatched_wandb_records"]))
