"""Reuse baseline environment checks; run real tuning smoke updates per role cell."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_scripts.baseline import preflight as base
from agents.common.tuning_plan import RESULTS,load_plan,candidate_config
from agents.common.tuning_records import inspect_trial
from agents.common.tuning_io import safe_root


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=('a','b','c'),required=True)
    p.add_argument('--device-policy',choices=('plan','cuda','cpu'),default='plan')
    p.add_argument('--wandb-mode',choices=('online','offline','disabled'),default='offline')
    p.add_argument('--results-root',type=Path,default=RESULTS)
    p.add_argument('--learning-check',action='store_true')
    p.add_argument('--mode',choices=('full','smoke'),default='full',help='resource envelope; learning trials always smoke')
    a=p.parse_args(argv)
    root=safe_root(a.results_root)
    rep=base.Report()
    base.check_python(rep)
    base.check_packages(rep)
    base.check_revision(rep)
    base.check_device(rep,a.role,a.device_policy)
    base.check_resources(rep,'memoryless' if a.role=='c' else 'implicit',str(root))
    if a.mode=='full':
        import shutil
        # Completed trials retain restart/replay plus final policy, so capacity must
        # cover the whole role, not only the peak live learner used by baseline preflight.
        needed={'a':60.,'b':70.,'c':220.}[a.role]
        free=shutil.disk_usage(root).free/1e9
        rep.add('retained tuning checkpoints',free>=needed,
                f'{free:.1f} GB free; conservative cumulative role allowance {needed:.0f} GB')
    base.check_wandb(rep,a.wandb_mode,'new_experiments',None,group='tuning-v3-preflight')
    plan=load_plan()
    rep.add('trial list',True,plan['plan_hash'])
    if a.learning_check:
        for cell in plan['cells']:
            if a.role not in cell['slot_roles']:
                continue
            cfg=candidate_config(plan,cell['env'],cell['algo'],0,101,'smoke')
            device='cuda' if a.device_policy=='cuda' or a.device_policy=='plan' and cell['env']=='colour' else 'cpu'
            # Separate root keeps repeated preflight attempts out of the smoke study.
            import tempfile
            with tempfile.TemporaryDirectory(prefix='pf',dir=root) as temp:
                cmd=[sys.executable,'-B',str(ROOT/'run_scripts/tuning/run_trial.py'),'--env',cell['env'],'--algo',cell['algo'],
                     '--candidate','0','--seed','101','--mode','smoke','--device',device,'--wandb-mode',a.wandb_mode,'--results-root',temp]
                code=subprocess.call(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                record=inspect_trial(temp,cfg,plan['plan_hash'])
                rep.add(f"learning {cell['env']}/{cell['algo']}",code==0 and record['state']=='complete',str(record))
    return int(rep.failed)


if __name__=='__main__':
    raise SystemExit(main())
