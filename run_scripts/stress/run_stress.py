"""Execute one registered stress run through the shared learner/recording pipeline.

    python -B run_scripts/stress/run_stress.py --env delivery --algo dqn --level default --seed 301 \
        --threads 1 [--mode smoke] [--wandb-mode offline] [--resume RUN_DIR]

The run must be in the frozen plan (stress_plan_v3.json, checked against the generator on
every start). --threads is required: it is passed to torch and recorded in metadata.json.
Colour runs require CUDA in full mode.
"""
import argparse
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.common.env_factory import STRESS_LEVELS  # noqa: E402
from agents.common.run_io import read_json  # noqa: E402
from agents.common.stress_io import StressDirectory, run_parent, safe_root  # noqa: E402
from agents.common.stress_plan import RESULTS, load_plan, run_config  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True)
    p.add_argument("--algo", required=True)
    p.add_argument("--level", required=True, choices=STRESS_LEVELS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", choices=("full", "smoke"), default="full")
    p.add_argument("--results-root", type=Path, default=RESULTS)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None,
                   help="default: the plan's device policy (colour cuda, others cpu)")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--resume")
    p.add_argument("--batch-id", default="manual")
    p.add_argument("--launcher-stdout")
    p.add_argument("--launcher-stderr")
    p.add_argument("--debug-fail-at-step", type=int)
    a = p.parse_args(argv)
    if a.threads < 1:
        p.error("--threads must be >= 1")
    plan = load_plan()
    try:
        entry, cfg = run_config(plan, a.env, a.algo, a.level, a.seed, a.mode)
    except ValueError as exc:
        p.error(str(exc))
    a.results_root = str(safe_root(a.results_root))
    a.device = a.device or entry["device"]
    a.machine_role = entry["role"]
    a.require_gpu = a.mode == "full" and a.env == "colour"
    a.stress_context = dict(plan_hash=plan["plan_hash"], level=a.level, role=entry["role"], slot=entry["slot"],
                            round=entry["round"], tuned_candidate=entry["candidate"])
    if a.resume:
        path = Path(a.resume).resolve()
        if path.parent != run_parent(a.results_root, cfg) or \
                read_json(str(path / "identity.json")).get("plan_hash") != plan["plan_hash"]:
            p.error("resume must use this run and plan inside the stress results root")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # Trust only this checkout for subprocess Git metadata (no global Git configuration is written).
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    os.environ.update(GIT_CONFIG_COUNT=str(count + 1))
    os.environ[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    os.environ[f"GIT_CONFIG_VALUE_{count}"] = ROOT.as_posix()
    from agents.common.runner import run, _on_sigterm
    signal.signal(signal.SIGTERM, _on_sigterm)
    return run(cfg, a, directory_class=StressDirectory)


if __name__ == "__main__":
    raise SystemExit(main())
