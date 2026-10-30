"""Stress-run directories: <root>/v3/<mode>/<env>/<algo>/<level>/sNNN/aNN/ (decision D11).

Reuses the tuning directory behaviour (attempt folders that never overwrite each other,
Windows-safe restart swaps, preserved resume evidence, W&B cache outside run paths) with
the stress CSV schema, which adds the per-episode required count and within-4,000-step
metrics.
"""
import os
from pathlib import Path

from agents.common.run_io import STRESS_CSV_FIELDS, new_run_uid
from agents.common.stress_plan import RESULTS, ROOT
from agents.common.tuning_io import TuningDirectory, check_path


def safe_root(root):
    root = Path(root or RESULTS).resolve()
    protected = (ROOT / "run_scripts/baseline", ROOT / "run_scripts/tuning", ROOT / "analysis",
                 ROOT / "results/v2", ROOT / "validation")
    if any(root == p or p in root.parents for p in protected):
        raise ValueError("stress runs cannot write inside baseline, tuning or analysis artifacts")
    return root


def run_parent(root, cfg):
    return (safe_root(root) / "v3" / cfg["mode"] / cfg["env"]["key"] / cfg["algo"]["name"]
            / cfg["stress"]["level"] / f"s{cfg['seed']['train_seed']}")


class StressDirectory(TuningDirectory):
    csv_fields = STRESS_CSV_FIELDS

    def swap_restart(self, tmp):
        """Atomic restart swap with a longer backoff than tuning's (about 50 s in total).

        On Windows a just-written replay file can stay locked by a virus scanner for several
        seconds (seen in the laptop rehearsal with the 5 MB SB3 smoke replay). Linux machines do
        not lock, so production behaviour there is unchanged.
        """
        import shutil
        import time
        old = self.restart_dir + ".old"

        def retry(fn, *args):
            for attempt in range(12):
                try:
                    return fn(*args)
                except PermissionError:
                    if attempt == 11:
                        raise
                    time.sleep(0.25 * (attempt + 1))
        if os.path.exists(old):
            retry(shutil.rmtree, old)
        if os.path.exists(self.restart_dir):
            retry(os.replace, self.restart_dir, old)
        try:
            retry(os.replace, tmp, self.restart_dir)
        except OSError:
            if os.path.exists(old) and not os.path.exists(self.restart_dir):
                retry(os.replace, old, self.restart_dir)
            raise
        if os.path.exists(old):
            retry(shutil.rmtree, old)

    @classmethod
    def create(cls, cfg, cfg_hash, results_root=None):
        parent = run_parent(results_root, cfg)
        for attempt in range(100):
            path = check_path(parent / f"a{attempt:02}", reserve=60)
            try:
                path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError("attempt limit exceeded")
        rd = cls(str(path))
        os.makedirs(rd.ckpt_dir)
        os.makedirs(rd.log_dir)
        rd.run_uid = new_run_uid()
        return rd
