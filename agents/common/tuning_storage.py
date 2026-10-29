"""Discard restart data only after independent verification of this attempt."""
from pathlib import Path
import shutil
from agents.common.tuning_io import safe_root, trial_parent
from agents.common.tuning_records import inspect_trial

DISK_GB = {'a': 20., 'b': 20., 'c': 25.}

def prune_completed(root, cfg, plan_hash, choices=None):
    root = safe_root(root)
    record = inspect_trial(root, cfg, plan_hash, choices)
    if record['state'] != 'complete':
        return False
    attempt = (root / record['run_dir']).resolve()
    if not attempt.is_relative_to(root) or attempt.parent != trial_parent(root, cfg).resolve():
        raise ValueError('attempt outside the registered trial')
    checkpoints = attempt / 'checkpoints'
    if checkpoints.is_symlink():
        raise ValueError('unsafe checkpoints directory')
    for name in ('restart', 'restart.old', 'restart.tmp'):
        target = checkpoints / name
        if target.is_symlink() or target.resolve().parent != checkpoints.resolve():
            raise ValueError('unsafe restart path')
        if target.exists():
            shutil.rmtree(target)
    return True
