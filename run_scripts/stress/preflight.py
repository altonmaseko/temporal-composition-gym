"""Preflight for one stress machine role. Nothing should be launched unless it prints PREFLIGHT PASSED.

    python -B run_scripts/stress/preflight.py --role a --threads 1 --wandb-mode online

Checks: interpreter and code source; packages against requirements.txt; code revision and the
baseline plan file; CUDA, VRAM and a GPU matmul (every role has colour runs); RAM and free disk
against STRESS_PROTOCOL.md section 11 (full mode: >= 16 GB RAM total, >= 35 GB free disk);
W&B login and an online run (online mode); the stress plan rebuilt on this machine's CPU and
compared field by field with the frozen file; --threads equal to the plan; and a tiny smoke
training of one colour cell of this role (CUDA unless --device-policy cpu), verified like a
production run, in a temporary results folder that is deleted afterwards.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_scripts.baseline import preflight as base  # noqa: E402

NEEDS = {"full": {"ram_total_gb": 16.0, "ram_avail_gb": 8.0, "disk_gb": 35.0},
         "smoke": {"ram_total_gb": 8.0, "ram_avail_gb": 2.0, "disk_gb": 2.0}}


def check_resources(rep, mode, root):
    import psutil
    need = NEEDS[mode]
    vm = psutil.virtual_memory()
    rep.add("ram", vm.total / 1e9 >= need["ram_total_gb"] and vm.available / 1e9 >= need["ram_avail_gb"],
            f"{vm.total / 1e9:.1f} GB total, {vm.available / 1e9:.1f} GB available (need >= {need['ram_total_gb']:.0f} GB "
            f"total and >= {need['ram_avail_gb']:.0f} GB available; peak colour process about 6 GB)")
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 1e9
    rep.add("disk", free >= need["disk_gb"], f"{free:.1f} GB free at {root} (need >= {need['disk_gb']:.0f} GB: "
            "restart data of one run, policy snapshots, software)")
    rep.add("cpu", True, f"{os.cpu_count()} logical CPUs; OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")


def check_plan(rep, threads, mode):
    import json
    from agents.common.stress_plan import PLAN_PATH, build_plan, json_diff
    try:
        rebuilt = build_plan()
        on_disk = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        diff = json_diff(on_disk, rebuilt)
    except Exception as exc:                                   # noqa: BLE001
        rep.add("stress plan on this CPU", False, f"could not rebuild: {exc!r}")
        return None
    rep.add("stress plan on this CPU", not diff,
            f"rebuilt plan equals the frozen file, hash {rebuilt['plan_hash']}" if not diff else
            "differs from the frozen file: " + "; ".join(diff[:8]))
    ok = threads == rebuilt["threads"]
    rep.add("threads", ok or mode == "smoke", f"--threads {threads}; plan value {rebuilt['threads']} (P16)",
            level="error" if mode == "full" else "warn")
    return rebuilt if not diff else None


def learning_check(rep, plan, role, device_policy, wandb_mode, threads):
    from agents.common.stress_plan import RESULTS, run_config
    from agents.common.stress_records import inspect_run
    job = next(r for r in sorted(plan["runs"], key=lambda r: r["slot"]) if r["role"] == role and r["env"] == "colour")
    device = "cpu" if device_policy == "cpu" else "cuda"
    entry, cfg = run_config(plan, job["env"], job["algo"], job["level"], job["seed"], "smoke")
    # Always directly under run_scripts/stress/results (git-ignored pf*): the W&B cache below it needs
    # 90 characters of headroom, which a deeper --results-root would not leave on Windows.
    RESULTS.mkdir(parents=True, exist_ok=True)
    temp = tempfile.mkdtemp(prefix="pf", dir=RESULTS)
    try:
        cmd = [sys.executable, "-B", str(ROOT / "run_scripts/stress/run_stress.py"), "--env", job["env"],
               "--algo", job["algo"], "--level", job["level"], "--seed", str(job["seed"]), "--mode", "smoke",
               "--device", device, "--wandb-mode", wandb_mode, "--threads", str(threads), "--results-root", temp]
        t = time.time()
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(threads)})
        rec = inspect_run(temp, cfg, plan["plan_hash"], entry)
        ok = res.returncode == 0 and rec["state"] == "complete"
        detail = (f"{job['env']}/{job['algo']}/{job['level']}/s{job['seed']} smoke on {device}: "
                  f"{time.time() - t:.0f}s, state {rec['state']}, updates {rec.get('n_updates')}")
        if not ok:
            tail = (res.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            detail += f"; exit {res.returncode}: {tail}; problems {[p for at in rec['attempts'] for p in at['problems']]}"
        rep.add("learning check", ok, detail)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def main(argv=None):
    from agents.common.stress_io import safe_root
    from agents.common.stress_plan import RESULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", choices=("a", "b", "c"), required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--mode", choices=("full", "smoke"), default="full", help="resource thresholds to apply")
    p.add_argument("--device-policy", choices=("plan", "cpu"), default="plan")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--results-root", type=Path, default=RESULTS)
    p.add_argument("--skip-learning-check", action="store_true")
    a = p.parse_args(argv)
    if a.mode == "full" and a.device_policy != "plan":
        p.error("full mode requires the plan's device policy (colour on CUDA)")
    root = safe_root(a.results_root)
    rep = base.Report()
    base.check_python(rep)
    base.check_packages(rep)
    base.check_revision(rep)
    base.check_device(rep, a.role, a.device_policy)
    check_resources(rep, a.mode, root)
    base.check_wandb(rep, a.wandb_mode, "new_experiments", None, group="stress-v3-preflight")
    plan = check_plan(rep, a.threads, a.mode)
    if plan and not a.skip_learning_check:
        learning_check(rep, plan, a.role, a.device_policy, a.wandb_mode, a.threads)
    print("PREFLIGHT FAILED" if rep.failed else "PREFLIGHT PASSED", flush=True)
    return int(rep.failed)


if __name__ == "__main__":
    raise SystemExit(main())
