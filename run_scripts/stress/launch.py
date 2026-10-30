"""Run one machine role's stress queue, one process at a time (decision D11).

The bash wrappers (run_a_stress.sh ...) choose the interpreter. Every run gets the plan's
--threads value and OMP/MKL thread variables. Colour runs use CUDA and fail rather than
fall back to the CPU; delivery and maintenance run on the CPU. With --continue-batch,
independently verified runs are skipped and an unfinished attempt with a restart checkpoint
is resumed. Restart/replay data is deleted only after a run passes independent verification.

Banners: BATCH COMPLETE (exit 0), BATCH FAILED (exit 1, at least one run failed),
PAUSED (exit 130, interrupted; rerun with --continue-batch).
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.common.run_io import write_json_atomic  # noqa: E402
from agents.common.stress_io import safe_root  # noqa: E402
from agents.common.stress_plan import RESULTS, load_plan, planning_hours, run_config  # noqa: E402
from agents.common.stress_records import inspect_run, prune_completed  # noqa: E402
from agents.common.tuning_io import check_path  # noqa: E402


def run_name(r):
    return f"{r['env']}/{r['algo']}/{r['level']}/s{r['seed']}"


def device_for(entry, policy):
    if policy == "cpu":
        return "cpu"
    return entry["device"]


class Estimate:
    """Planned hours for the role, rescaled by how fast this machine has actually been."""

    def __init__(self, jobs):
        self.planned = {run_name(j): planning_hours(j["env"], j["algo"], j["level"]) for j in jobs}
        self.done_planned = 0.0
        self.done_actual = 0.0

    def record(self, job, seconds):
        self.done_planned += self.planned[run_name(job)]
        self.done_actual += seconds / 3600

    def factor(self):
        return self.done_actual / self.done_planned if self.done_planned > 0.05 else 1.0

    def remaining(self, jobs):
        return sum(self.planned[run_name(j)] for j in jobs) * self.factor()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", required=True, choices=("a", "b", "c"))
    p.add_argument("--mode", choices=("full", "smoke"), default="full")
    p.add_argument("--threads", type=int, required=True, help="must equal the plan's value in full mode")
    p.add_argument("--results-root", type=Path, default=RESULTS)
    p.add_argument("--device-policy", choices=("plan", "cpu"), default="plan",
                   help="cpu is allowed in smoke mode only (laptop rehearsal)")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    p.add_argument("--continue-batch", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prune-completed-restarts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--debug-fail-run", metavar="ENV/ALGO/LEVEL/sSEED@STEP",
                   help="smoke rehearsal only: make one run fail at this env step")
    a = p.parse_args(argv)
    plan = load_plan()
    if a.mode == "full":
        if a.threads != plan["threads"]:
            p.error(f"--threads {a.threads} differs from the frozen plan ({plan['threads']}, decision P16)")
        if a.device_policy != "plan":
            p.error("full mode uses the plan's device policy (colour on CUDA)")
        if a.debug_fail_run:
            p.error("--debug-fail-run is for smoke rehearsals only")
    fail_name, fail_step = (a.debug_fail_run.split("@") + [None])[:2] if a.debug_fail_run else (None, None)
    root = safe_root(a.results_root)
    jobs = sorted((r for r in plan["runs"] if r["role"] == a.role), key=lambda r: r["slot"])
    if not a.dry_run and a.device_policy == "plan" and any(j["device"] == "cuda" for j in jobs):
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available but this role has colour runs that require it; refusing to start.",
                  file=sys.stderr, flush=True)
            print("BATCH FAILED", flush=True)
            return 1
    batch = datetime.now(timezone.utc).strftime("%m%d%H%M%S") + "-" + str(os.getpid())
    folder = check_path(root / "v3" / a.mode / "b" / a.role / batch, reserve=40)
    if not a.dry_run:
        folder.mkdir(parents=True)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               WANDB_MODE=a.wandb_mode)
    est = Estimate(jobs)
    print(f"role {a.role}: {len(jobs)} runs, plan {plan['plan_hash'][:12]}, mode {a.mode}, threads {a.threads}", flush=True)
    print(f"ESTIMATE at launch: {est.remaining(jobs):.1f} h planned for this role "
          f"(planning reference{'; smoke runs are much shorter' if a.mode == 'smoke' else ''})", flush=True)
    failures, started_all = 0, time.monotonic()
    for i, job in enumerate(jobs):
        name = run_name(job)
        entry, cfg = run_config(plan, job["env"], job["algo"], job["level"], job["seed"], a.mode)
        record = inspect_run(root, cfg, plan["plan_hash"], entry)
        cmd = [sys.executable, "-B", str(ROOT / "run_scripts/stress/run_stress.py"), "--env", job["env"],
               "--algo", job["algo"], "--level", job["level"], "--seed", str(job["seed"]), "--mode", a.mode,
               "--results-root", str(root), "--device", device_for(job, a.device_policy),
               "--wandb-mode", a.wandb_mode, "--threads", str(a.threads), "--batch-id", batch]
        state, code = "pending", 0
        if record["state"] == "complete":
            state = "skipped_complete" if a.continue_batch else "existing_requires_continue"
            code = 0 if a.continue_batch else 1
        elif record["state"] == "duplicate":
            state, code = "ambiguous_attempts", 1
        elif record["attempts"] and not a.continue_batch:
            state, code = "existing_requires_continue", 1
        elif record["attempts"]:
            last = record["attempts"][-1]
            path = root / last["run_dir"]
            if (path / "checkpoints/restart/state.json").exists():
                cmd += ["--resume", str(path)]
        if name == fail_name and state == "pending" and "--resume" not in cmd:
            cmd += ["--debug-fail-at-step", fail_step]
        if a.dry_run:
            print(f"{i:03d} {name} {state} {subprocess.list2cmdline(cmd)}", flush=True)
            continue
        started = time.monotonic()
        if state == "pending":
            out, err = folder / f"{i:03d}.out.log", folder / f"{i:03d}.err.log"
            cmd += ["--launcher-stdout", str(out), "--launcher-stderr", str(err)]
            print(f"START {i + 1}/{len(jobs)} {name}" + (" (resume)" if "--resume" in cmd else ""), flush=True)
            with out.open("w") as stdout, err.open("w") as stderr:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                try:
                    code = proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    code = 130
            verified = inspect_run(root, cfg, plan["plan_hash"], entry)["state"] == "complete"
            state = "complete" if code == 0 and verified else "failed"
            if state == "failed":
                code = code or 1
            else:
                est.record(job, time.monotonic() - started)
        if a.prune_completed_restarts and state in ("complete", "skipped_complete"):
            prune_completed(root, cfg, plan["plan_hash"], entry)
        seconds = time.monotonic() - started
        with (folder / "status.csv").open("a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["slot", "run", "state", "exit_code", "seconds"])
            w.writerow([job["slot"], name, state, code, round(seconds, 1)])
        failures += bool(code)
        left = jobs[i + 1:]
        remaining = est.remaining(left)
        progress = {"role": a.role, "mode": a.mode, "batch": batch, "plan_hash": plan["plan_hash"],
                    "done": i + 1, "total": len(jobs), "last_run": name, "last_state": state, "failures": failures,
                    "elapsed_h": round((time.monotonic() - started_all) / 3600, 3),
                    "remaining_h_estimate": round(remaining, 2), "speed_factor": round(est.factor(), 3),
                    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        write_json_atomic(str(folder / "progress.json"), progress)
        print(f"{state.upper()} {name} ({seconds / 60:.1f} min). ESTIMATE: {i + 1}/{len(jobs)} done, "
              f"~{remaining:.1f} h left (planned x {est.factor():.2f})", flush=True)
        if code == 130:
            print("PAUSED: interrupted. Rerun the same command with --continue-batch to resume. "
                  "This program cannot stop billing; stop or destroy the machine yourself.", flush=True)
            return 130
    print("BATCH FAILED" if failures else "BATCH COMPLETE", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("PAUSED: interrupted between runs. Rerun the same command with --continue-batch.", flush=True)
        raise SystemExit(130)
