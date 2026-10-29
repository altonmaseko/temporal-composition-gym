"""Execute one registered trial through the shared learner/recording pipeline."""
import argparse
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.common.tuning_plan import RESULTS, load_plan, candidate_config, slots
from agents.common.tuning_io import TuningDirectory, safe_root, trial_parent
from agents.common.tuning_records import load_finalists
from agents.common.run_io import read_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('env', 'algo'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--candidate', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--mode', choices=('full','smoke'), default='full')
    p.add_argument('--results-root', type=Path, default=RESULTS)
    p.add_argument('--device', choices=('cpu','cuda','auto'), default='cpu')
    p.add_argument('--wandb-mode', choices=('online','offline','disabled'), default='offline')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--resume')
    p.add_argument('--batch-id', default='manual')
    p.add_argument('--launcher-stdout')
    p.add_argument('--launcher-stderr')
    p.add_argument('--debug-fail-at-step', type=int)
    a = p.parse_args(argv)
    plan = load_plan()
    a.results_root = str(safe_root(a.results_root))
    assignment = load_finalists(a.results_root, plan, a.mode)
    match = [s for s in slots(plan, a.mode, assignment['finalists'] if assignment else None)
             if (s['env'],s['algo'],s['candidate'],s['seed']) == (a.env,a.algo,a.candidate,a.seed)]
    if len(match) != 1:
        p.error('trial is not registered (confirmation requires frozen finalists.json)')
    slot = match[0]
    cfg = candidate_config(plan,a.env,a.algo,a.candidate,a.seed,a.mode)
    a.machine_role = slot['role']
    a.require_gpu = a.mode == 'full' and a.env == 'colour'
    a.tuning_context = dict(plan_hash=plan['plan_hash'], candidate=a.candidate, stage=slot['stage'], slot=slot['slot'],
                            assignment_hash=assignment['assignment_hash'] if a.seed != 101 else None)
    if a.resume:
        path = Path(a.resume).resolve()
        if path.parent != trial_parent(a.results_root,cfg) or read_json(str(path/'identity.json')).get('plan_hash') != plan['plan_hash']:
            p.error('resume must use this trial and plan inside the tuning root')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    # Trust only this explicitly selected checkout for subprocess Git metadata;
    # no global Git configuration is written (Windows sandbox ownership differs).
    count = int(os.environ.get('GIT_CONFIG_COUNT', '0'))
    os.environ.update(GIT_CONFIG_COUNT=str(count+1))
    os.environ[f'GIT_CONFIG_KEY_{count}'] = 'safe.directory'
    os.environ[f'GIT_CONFIG_VALUE_{count}'] = ROOT.as_posix()
    from agents.common.runner import run, _on_sigterm
    signal.signal(signal.SIGTERM, _on_sigterm)
    return run(cfg,a,directory_class=TuningDirectory)


if __name__ == '__main__':
    raise SystemExit(main())
