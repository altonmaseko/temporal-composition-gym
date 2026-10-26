"""DQN (SB3), memoryless baseline for Hazardous Delivery and Sequential Colour.
With --task-state: DQN-TS control (observation + reward-machine task state).

Protocol-v2 baseline entry point. All behaviour (configuration, run directory, W&B,
evaluation, checkpoints, explicit resume) lives in agents/common/runner.py; the run's
settings come from agents/common/baseline_plan.py.

Examples (from any working directory):
    python agents/train_dqn.py --env delivery --seed 1                    # full run, W&B online
    python agents/train_dqn.py --env delivery --seed 1 --mode smoke       # short labelled pipeline check
    python agents/train_dqn.py --env delivery --seed 1 --print-config     # show the resolved config + hash
    python agents/train_dqn.py --env delivery --seed 1 --resume RUN_DIR   # continue a specific run
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.common.runner import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli("dqn", __doc__, task_state_flag=True))
