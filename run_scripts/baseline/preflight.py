"""Resource and environment preflight for one baseline machine (protocol v2).

Checks, without installing anything:
  - the selected interpreter and that this checkout's code is the one imported
  - required packages and their versions against requirements.txt
  - CUDA / GPU and VRAM (required when the role's colour runs use the GPU)
  - RAM and free disk for the role's largest run (image replay + restart checkpoints)
  - W&B: logged in (via W&B's own login, e.g. ~/.netrc from `wandb login`) and able to
    create an ONLINE run in the configured project. The API key is never read here.
  - the code revision, a clean checkout and an up-to-date baseline_plan.json
  - optionally (--learning-check) a short labelled smoke training of each of the role's
    algorithms: real gradient updates, evaluations, a restart checkpoint, the planned device,
    and (online mode) an online W&B run created by the training process; plus rough timings

    python run_scripts/baseline/preflight.py --role implicit --wandb-mode online --learning-check

Exit code 0 = all checks passed; 1 = at least one failed (nothing should be launched).
"""
import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Peak memory and disk per role, from the protocol's estimates (largest run of the role).
ROLE_NEEDS = {
    "memoryless": {"ram_gb": 8.0, "disk_gb": 15.0, "largest": "DQN colour: 4.2 GB replay (obs+next) + restart copy"},
    "implicit": {"ram_gb": 6.0, "disk_gb": 10.0, "largest": "DRQN colour: 2.3 GB frame replay incl. LSTM state"},
    "explicit": {"ram_gb": 6.0, "disk_gb": 10.0, "largest": "QRM-Cue colour: 2.1 GB frame replay"},
}
REQUIRED = ["torch", "stable_baselines3", "sb3_contrib", "gymnasium", "minigrid", "pymunk", "pygame", "numpy",
            "wandb", "psutil", "pytablericons", "PIL", "requests"]
DIST_NAMES = {"PIL": "pillow", "stable_baselines3": "stable_baselines3", "sb3_contrib": "sb3_contrib"}


class Report:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail, level="error"):
        status = "PASS" if ok else ("FAIL" if level == "error" else "WARN")
        self.items.append({"check": name, "status": status, "detail": detail})
        print(f"[{status}] {name}: {detail}", flush=True)

    @property
    def failed(self):
        return any(i["status"] == "FAIL" for i in self.items)


def pinned_versions():
    pins = {}
    path = os.path.join(REPO_ROOT, "requirements.txt")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "==" in line:
                name, ver = line.split("==", 1)
                pins[name.lower().replace("-", "_")] = ver
    return pins


def check_python(rep):
    ok = sys.version_info >= (3, 10)
    rep.add("python", ok, f"{sys.executable} ({sys.version.split()[0]})")
    import temporal_comp_gym
    import agents
    here = all(os.path.commonpath([os.path.abspath(m.__file__), REPO_ROOT]) == REPO_ROOT
               for m in (temporal_comp_gym, agents))
    rep.add("code source", here, f"temporal_comp_gym and agents imported from {REPO_ROOT}" if here else
            f"imported from {temporal_comp_gym.__file__} / {agents.__file__} (not this checkout)")


def check_packages(rep):
    import importlib
    import importlib.metadata as md
    pins = pinned_versions()
    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
        except Exception as exc:                                           # noqa: BLE001
            rep.add(f"package {mod}", False, f"import failed: {exc}")
            continue
        dist = DIST_NAMES.get(mod, mod)
        try:
            ver = md.version(dist)
        except md.PackageNotFoundError:
            ver = None
        pin = pins.get(dist.lower().replace("-", "_"))
        same = pin is None or ver is None or ver.split("+")[0] == pin
        rep.add(f"package {mod}", True if same else False, f"{ver} (pinned {pin})" if pin else f"{ver}",
                level="error" if same else "warn")
    try:
        md.version("pygame-ce")
        rep.add("pygame-ce", False, "pygame-ce is installed alongside pygame (same import name)", level="warn")
    except md.PackageNotFoundError:
        pass


def check_device(rep, role, device_policy):
    import torch
    need_gpu = device_policy in ("plan", "cuda")          # every role has a colour (image) run
    if not torch.cuda.is_available():
        rep.add("cuda", not need_gpu, "CUDA not available" + (" but the role's colour runs require a GPU"
                                                             if need_gpu else " (device policy: cpu)"))
        return
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 1e9
    rep.add("cuda", True, f"{torch.cuda.device_count()} GPU(s); using cuda:0 = {p.name}, {vram:.1f} GB VRAM, "
            f"CUDA {torch.version.cuda}")
    rep.add("vram", vram >= 4.0, f"{vram:.1f} GB (need >= 4 GB)")
    x = torch.randn(256, 256, device="cuda")
    rep.add("cuda compute", bool(torch.isfinite((x @ x).sum()).item()), "matmul on GPU ok")


def check_resources(rep, role, results_root):
    import psutil
    need = ROLE_NEEDS[role]
    vm = psutil.virtual_memory()
    rep.add("ram", vm.available / 1e9 >= need["ram_gb"],
            f"{vm.available / 1e9:.1f} GB available of {vm.total / 1e9:.1f} GB (need >= {need['ram_gb']} GB; "
            f"largest run: {need['largest']})")
    os.makedirs(results_root, exist_ok=True)
    free = shutil.disk_usage(results_root).free / 1e9
    rep.add("disk", free >= need["disk_gb"], f"{free:.1f} GB free at {results_root} (need >= {need['disk_gb']} GB)")
    rep.add("cpu", True, f"{os.cpu_count()} logical CPUs; OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")


def check_wandb(rep, mode, project, entity):
    from agents.common.baseline_plan import FORBIDDEN_WANDB_PROJECTS
    if project in FORBIDDEN_WANDB_PROJECTS:
        rep.add("wandb project", False, f"{project!r} is the pilot project")
        return
    if mode != "online":
        rep.add("wandb", False, f"mode={mode}: online logging not checked", level="warn")
        return
    import wandb
    try:
        ok = wandb.login(anonymous="never", relogin=False, verify=True)
    except Exception as exc:                                               # noqa: BLE001
        ok, err = False, exc
    else:
        err = None
    if not ok:
        rep.add("wandb login", False, f"not logged in ({err!r}). Run `wandb login` on this machine first.")
        return
    rep.add("wandb login", True, "logged in (credentials handled by W&B; not read by this code)")
    try:
        run = wandb.init(project=project, entity=entity, mode="online", group="preflight-v2", job_type="preflight",
                         name=f"preflight-{socket.gethostname()}-{time.strftime('%Y%m%dT%H%M%S')}",
                         config={"purpose": "baseline preflight"})
        run.log({"preflight/ok": 1})
        url = run.url
        run.finish()
        rep.add("wandb online run", True, f"created {url} in project {project!r}")
    except Exception as exc:                                               # noqa: BLE001
        rep.add("wandb online run", False, f"could not create an online run in {project!r}: {exc!r}")


def check_revision(rep):
    from agents.common.baseline_plan import main as plan_main
    from agents.common.run_io import git_info
    g = git_info()
    rep.add("code revision", bool(g["rev"]), f"{g['rev']} on {g['branch']}")
    rep.add("clean checkout", g["dirty"] is False,
            "no uncommitted changes to tracked files" if g["dirty"] is False else
            f"uncommitted changes: {g['changed_tracked_files'][:6]}", level="warn")
    rep.add("baseline plan file", plan_main(["write", "--check"]) == 0, "run_scripts/baseline/baseline_plan.json matches the code")


def _run_dirs(root):
    import glob
    return set(glob.glob(os.path.join(root, "v2", "smoke", "*", "*", "seed*__*")))


def learning_check(rep, role, device_policy, wandb_mode, project, entity, results_root):
    """Short labelled smoke training of every algorithm of the role (results/v2/smoke).

    Passing means: exit 0, verified budget, gradient updates > 0, every evaluation point and
    the final evaluation written, a restart checkpoint saved, the planned device used, and
    (online mode) a W&B URL created by the training process itself.
    """
    import csv
    from agents.common.baseline_plan import ALGORITHMS, BUDGETS, ROLE_CONTROL_RUNS, ROLE_MAIN_RUNS
    from agents.common.run_io import read_json
    projections = {}
    for env_key, algo in ROLE_MAIN_RUNS[role] + ROLE_CONTROL_RUNS[role]:
        dev = "cpu" if device_policy == "cpu" or (device_policy == "plan" and env_key != "colour") else "cuda"
        cmd = [sys.executable, os.path.join(REPO_ROOT, ALGORITHMS[algo]["script"]), "--env", env_key, "--seed", "1",
               "--mode", "smoke", "--device", dev, "--wandb-mode", wandb_mode, "--wandb-project", project,
               "--results-root", results_root]
        if dev == "cuda":
            cmd.append("--require-gpu")
        if algo == "dqn_ts":
            cmd.append("--task-state")
        if entity:
            cmd += ["--wandb-entity", entity]
        before = _run_dirs(results_root)
        t = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        name = f"learning check {env_key}/{algo}"
        new = sorted(_run_dirs(results_root) - before)
        if res.returncode != 0 or len(new) != 1:
            err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "no run directory"
            rep.add(name, False, f"exit {res.returncode}: {err}")
            continue
        rd = new[0]
        st = read_json(os.path.join(rd, "status.json"))
        meta = read_json(os.path.join(rd, "metadata.json"))
        with open(os.path.join(rd, "eval_points.csv"), encoding="utf-8") as f:
            points = list(csv.DictReader(f))
        with open(os.path.join(rd, "resources.csv"), encoding="utf-8") as f:
            res_rows = list(csv.DictReader(f))
        problems = []
        if not (st.get("state") == "complete" and st.get("verified")):
            problems.append(f"status {st.get('state')}")
        if not st.get("n_updates"):
            problems.append("no gradient updates")
        if len(points) < 2:
            problems.append("fewer than 2 evaluation points")
        if not os.path.exists(os.path.join(rd, "checkpoints", "restart", "state.json")):
            problems.append("no restart checkpoint")
        if dev == "cuda" and not meta["device"].get("is_gpu_run"):
            problems.append("did not run on the GPU")
        if wandb_mode == "online" and not (st.get("wandb") or {}).get("url"):
            problems.append("no online W&B URL from the training process")
        sps = [float(r["env_steps_per_s"]) for r in res_rows if r.get("env_steps_per_s")]
        gpu = max([float(r["gpu_mem_max_gb"]) for r in res_rows if r.get("gpu_mem_max_gb")] or [0.0])
        rss = max([float(r["rss_gb"]) for r in res_rows if r.get("rss_gb")] or [0.0])
        if sps:
            # Rough: smoke throughput (incl. its short evaluations) applied to the full budget.
            projections[f"{env_key}/{algo}"] = round(BUDGETS[env_key] / sps[-1] / 3600, 2)
        rep.add(name, not problems,
                f"{time.time() - t:.0f}s, updates={st.get('n_updates')}, device={meta['device']['selected']}, "
                f"peak RSS {rss:.2f} GB, peak GPU {gpu:.2f} GB, {sps[-1] if sps else float('nan'):.0f} env steps/s"
                + (f", W&B {st['wandb']['url']}" if (st.get("wandb") or {}).get("url") else "")
                + (f" -- problems: {problems}" if problems else ""))
    if projections:
        total = sum(projections.values()) * 5
        rep.add("rough full-run time", False,
                f"hours per run (smoke throughput x full budget): {projections}; x5 seeds ~ {total:.0f} h "
                "(lower bound: full-size replay/batches and full evaluations are slower)", level="warn")
    return projections


def main():
    from agents.common.baseline_plan import DEFAULT_WANDB_PROJECT, ROLES
    from agents.common.run_io import DEFAULT_RESULTS_ROOT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", choices=ROLES, required=True)
    p.add_argument("--mode", choices=("full", "smoke"), default="full")
    p.add_argument("--device-policy", choices=("plan", "cuda", "cpu"), default="plan")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--learning-check", action="store_true", help="also run a short smoke training of each algorithm")
    p.add_argument("--out", default=None, help="write the report as JSON")
    args = p.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    rep = Report()
    print(f"preflight: role={args.role} host={socket.gethostname()} {platform.platform()}", flush=True)
    for fn in (lambda: check_python(rep), lambda: check_revision(rep), lambda: check_packages(rep),
               lambda: check_device(rep, args.role, args.device_policy),
               lambda: check_resources(rep, args.role, args.results_root),
               lambda: check_wandb(rep, args.wandb_mode, args.wandb_project, args.wandb_entity)):
        try:
            fn()
        except Exception as exc:                                           # noqa: BLE001
            rep.add("preflight internal", False, repr(exc))
    if args.learning_check and not rep.failed:
        learning_check(rep, args.role, args.device_policy, args.wandb_mode, args.wandb_project, args.wandb_entity,
                       args.results_root)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"role": args.role, "host": socket.gethostname(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "failed": rep.failed, "checks": rep.items}, f, indent=2)
    print("PREFLIGHT " + ("FAILED" if rep.failed else "PASSED"), flush=True)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
