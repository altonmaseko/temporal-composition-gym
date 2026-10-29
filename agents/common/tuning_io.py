"""Compact, isolated tuning artifacts using the shared CSV/status implementation."""
import os
from pathlib import Path

from agents.common.run_io import RunDirectory, new_run_uid, utc_now, write_json_atomic
from agents.common.tuning_plan import RESULTS, ROOT


def safe_root(root):
    root = Path(root or RESULTS).resolve()
    protected = (ROOT / 'run_scripts/baseline', ROOT / 'analysis', ROOT / 'results/v2', ROOT / 'validation')
    if any(root == p or p in root.parents for p in protected):
        raise ValueError('tuning cannot write inside baseline or analysis artifacts')
    return root


def check_path(path, reserve=0):
    if len(str(Path(path).absolute())) + reserve >= 240:
        raise ValueError(f'path exceeds portable 239-character limit: {path}')
    return Path(path)


def trial_parent(root, cfg):
    return safe_root(root) / 'v3' / cfg['mode'] / cfg['env']['key'] / cfg['algo']['name'] / f"t{cfg['tuning']['candidate']:02}" / f"s{cfg['seed']['train_seed']}"


class TuningDirectory(RunDirectory):
    def preserve_status(self, status):
        stamp = utc_now().strftime('%Y%m%dT%H%M%S%fZ')
        write_json_atomic(str(check_path(self.file(f'status_resume_{stamp}.json'))), status)

    def swap_restart(self, tmp):
        """Retry transient Windows file locks without changing baseline checkpoint IO."""
        import shutil
        import time
        old = self.restart_dir + '.old'
        def retry(fn, *args):
            for attempt in range(6):
                try:
                    return fn(*args)
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(.2 * (attempt + 1))
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
        parent = trial_parent(results_root, cfg)
        for attempt in range(10000):
            path = check_path(parent / f'a{attempt:02}', reserve=60)
            try:
                path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError('attempt limit exceeded')
        rd = cls(str(path))
        os.makedirs(rd.ckpt_dir)
        os.makedirs(rd.log_dir)
        rd.run_uid = new_run_uid()
        return rd

    @property
    def wandb_path(self):
        # External SDK caches have longer names; keep them outside trial paths.
        root = Path(self.path).parents[6]
        # Current SDK filenames need 88 characters below this directory. Reserve
        # 90, also allowing the preflight's temporary results-root component.
        path = check_path(root / '_wb' / self.path_hash, reserve=90)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def path_hash(self):
        import hashlib
        return hashlib.sha256(self.path.encode()).hexdigest()[:12]

    def truncate_records(self, max_step):
        super().truncate_records(max_step)
        # Both terminal evaluations are repeated after restart; preserve evidence.
        stamp = utc_now().strftime('%Y%m%dT%H%M%SZ')
        noisy = Path(self.file('eval_final_noisy.csv'))
        if noisy.exists():
            noisy.rename(check_path(self.file(f'eval_final_noisy.pre-resume-{stamp}.csv')))
        summary = Path(self.file('final_summary.json'))
        if summary.exists():
            summary.replace(check_path(self.file(f'final_summary.pre-resume-{stamp}.json')))
        # Preserve shared truncation backups in CSV-only collection bundles too.
        for backup in Path(self.path).glob('*.csv.pre-resume-*.bak'):
            backup.rename(check_path(backup.with_suffix('.csv')))
