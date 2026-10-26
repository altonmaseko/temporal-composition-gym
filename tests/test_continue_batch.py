"""--continue-batch restart procedure: skip complete runs, resume unfinished ones, start the rest fresh."""
import os
import subprocess
import sys

import pytest

from tests.test_pipeline import BASH, ENV, REPO, needs_bash, run_dirs, run_script

sys.path.insert(0, os.path.join(REPO, "run_scripts", "baseline"))
import batch_action  # noqa: E402


@pytest.mark.slow
def test_batch_action_fresh_resume_skip(tmp_path):
    root = str(tmp_path)
    assert batch_action.decide(root, "smoke", "delivery", "dqn", 1) == ("fresh", None)
    res = run_script("agents/train_dqn.py", "--env", "delivery", "--seed", "1", "--mode", "smoke",
                     "--wandb-mode", "disabled", "--debug-fail-at-step", "3000", root=tmp_path)
    assert res.returncode == 1
    (rd,) = run_dirs(tmp_path)
    assert batch_action.decide(root, "smoke", "delivery", "dqn", 1) == ("resume", rd)
    # an override run with a different config hash is never picked up for the planned run
    other = run_script("agents/train_dqn.py", "--env", "delivery", "--seed", "1", "--mode", "smoke",
                       "--wandb-mode", "disabled", "--set", "algo.hyperparams.learning_rate=0.001",
                       "--debug-fail-at-step", "2500", root=tmp_path)
    assert other.returncode == 1
    assert batch_action.decide(root, "smoke", "delivery", "dqn", 1) == ("resume", rd)
    done = run_script("agents/train_dqn.py", "--env", "delivery", "--seed", "1", "--mode", "smoke",
                      "--wandb-mode", "disabled", "--resume", rd, root=tmp_path)
    assert done.returncode == 0
    assert batch_action.decide(root, "smoke", "delivery", "dqn", 1) == ("skip", rd)
    assert batch_action.decide(root, "smoke", "delivery", "dqn", 2) == ("fresh", None)

    if BASH is None:
        return
    script = os.path.join(REPO, "run_scripts", "baseline", "run_memoryless_baselines.sh")
    out = subprocess.run([BASH, script, "--python", sys.executable.replace("\\", "/"), "--mode", "smoke", "--seeds", "1",
                          "--no-controls", "--results-root", root.replace("\\", "/"), "--continue-batch", "--dry-run"],
                         capture_output=True, text=True, env=ENV)
    assert out.returncode == 0, out.stderr
    assert "delivery / dqn / seed 1  skipped: already complete" in out.stdout
    cmds = [l for l in out.stdout.splitlines() if "train_" in l]
    assert len(cmds) == 2 and not any("--resume" in c for c in cmds)       # colour/dqn and maintenance/ppo fresh
