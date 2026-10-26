"""What the launcher should do for one planned run when continuing a batch (--continue-batch).

Prints one line, tab-separated:
  skip<TAB><run_dir>     a verified complete run with the planned config hash exists
  resume<TAB><run_dir>   the most recent unfinished run with that hash has a restart checkpoint
  fresh                  otherwise (start a new run)

    python run_scripts/baseline/batch_action.py --env colour --algo drqn --seed 3 [--mode full] [--root DIR]
"""
import argparse
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from agents.common import baseline_plan as plan  # noqa: E402
from agents.common.env_factory import PROTOCOL_VERSION  # noqa: E402
from agents.common.run_io import DEFAULT_RESULTS_ROOT, read_json  # noqa: E402
from check_manifest import verify_run  # noqa: E402


def decide(root, mode, env_key, algo, seed):
    expected = plan.config_hash(plan.resolve_config(env_key, algo, seed, mode))
    pattern = os.path.join(root, PROTOCOL_VERSION, mode, env_key, algo, f"seed{seed}__{expected[:10]}__*")
    candidates = []
    for rd in sorted(glob.glob(pattern)):
        try:
            ident = read_json(os.path.join(rd, "identity.json"))
        except (OSError, ValueError):
            continue
        if ident.get("config_hash") != expected:
            continue
        ok, _, _ = verify_run(rd)
        if ok:
            return "skip", rd
        status = read_json(os.path.join(rd, "status.json")) if os.path.exists(os.path.join(rd, "status.json")) else {}
        if status.get("state") != "complete" and os.path.exists(os.path.join(rd, "checkpoints", "restart", "state.json")):
            candidates.append((ident.get("created_utc", ""), rd))
    if candidates:
        return "resume", max(candidates)[1]
    return "fresh", None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True)
    p.add_argument("--algo", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", choices=("full", "smoke"), default="full")
    p.add_argument("--root", default=DEFAULT_RESULTS_ROOT)
    a = p.parse_args(argv)
    action, rd = decide(a.root, a.mode, a.env, a.algo, a.seed)
    print(action if rd is None else f"{action}\t{rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
