"""DRQN (custom), implicit-memory baseline for Hazardous Delivery and Sequential Colour.

Protocol-v2 baseline entry point. All behaviour (configuration, run directory, W&B,
evaluation, checkpoints, explicit resume) lives in agents/common/runner.py; the run's
settings come from agents/common/baseline_plan.py.

Examples (from any working directory):
    python agents/drqn/train_drqn.py --env colour --seed 1                    # full run, W&B online
    python agents/drqn/train_drqn.py --env colour --seed 1 --mode smoke       # short labelled pipeline check
    python agents/drqn/train_drqn.py --env colour --seed 1 --print-config     # show the resolved config + hash
    python agents/drqn/train_drqn.py --env colour --seed 1 --resume RUN_DIR   # continue a specific run
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.common.runner import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli("drqn", __doc__))
