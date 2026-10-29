"""Sequential, restartable role batches. Bash wrappers select the interpreter."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agents.common.tuning_plan import RESULTS,load_plan,slots,candidate_config
from agents.common.tuning_records import inspect_trial,load_finalists
from agents.common.tuning_io import safe_root,check_path
from agents.common.tuning_budget import remaining_seconds,projected_seconds
from agents.common.run_io import write_json_atomic


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',required=True,choices=('a','b','c','all'))
    p.add_argument('--stage',choices=('screen','confirm'),default='screen')
    p.add_argument('--mode',choices=('full','smoke'),default='full')
    p.add_argument('--results-root',type=Path,default=RESULTS)
    p.add_argument('--device-policy',choices=('plan','cpu','cuda'),default='plan')
    p.add_argument('--wandb-mode',choices=('online','offline','disabled'),default='offline')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--continue-batch',action='store_true')
    p.add_argument('--billing-start',help='UTC rental start, e.g. 2026-10-28T12:00:00Z; include setup/preflight')
    p.add_argument('--previously-billed-hours',type=float,default=0.,help='all earlier rental sessions for this role')
    p.add_argument('--hourly-rate',type=float,default=.30)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--attempt-choices',type=Path)
    a=p.parse_args(argv)
    if a.mode=='full' and a.role=='all':
        p.error('production requires a single role and its fixed share of the cap')
    if a.mode=='full' and not a.dry_run and not a.billing_start:
        p.error('--billing-start is required to enforce the agreed billed-hour/cost cap')
    root=safe_root(a.results_root)
    plan=load_plan()
    choices=json.loads(a.attempt_choices.read_text()) if a.attempt_choices else None
    assignment=load_finalists(root,plan,a.mode)
    if a.stage=='confirm' and not assignment:
        p.error('collect screening records and run select_configs.py --prepare-confirmation first')
    jobs=[s for s in slots(plan,a.mode,assignment['finalists'] if assignment else None)
          if s['stage']==a.stage and (a.role=='all' or s['role']==a.role)]
    batch=datetime.now(timezone.utc).strftime('%m%d%H%M%S')+'-'+str(os.getpid())
    folder=check_path(root/'v3'/a.mode/'b'/a.role/batch,reserve=40)
    if not a.dry_run:
        folder.mkdir(parents=True)
        if a.mode=='full':
            ledger=root/'v3'/a.mode/f'billing_{a.role}.json'
            terms=dict(billing_start=a.billing_start,previously_billed_hours=a.previously_billed_hours,hourly_rate=a.hourly_rate)
            if ledger.exists() and json.loads(ledger.read_text())!=terms:
                p.error('billing terms changed; preserve ledger and reconcile previous sessions before a new root/session')
            write_json_atomic(str(ledger),terms)
    failures=0
    observed={}
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(a.threads),WANDB_MODE=a.wandb_mode)
    for i,s in enumerate(jobs):
        cfg=candidate_config(plan,s['env'],s['algo'],s['candidate'],s['seed'],a.mode)
        record=inspect_trial(root,cfg,plan['plan_hash'],choices)
        name=f"{s['env']}/{s['algo']}/t{s['candidate']:02}/s{s['seed']}"
        state,code='pending',0
        cmd=[sys.executable,'-B',str(ROOT/'run_scripts/tuning/run_trial.py'),'--env',s['env'],'--algo',s['algo'],
             '--candidate',str(s['candidate']),'--seed',str(s['seed']),'--mode',a.mode,'--results-root',str(root),
             '--device','cuda' if a.device_policy=='cuda' or a.device_policy=='plan' and s['env']=='colour' else 'cpu',
             '--wandb-mode',a.wandb_mode,'--threads',str(a.threads),'--batch-id',batch]
        if a.continue_batch and record['state']=='complete':
            state='skipped_complete'
        elif record['state']!='missing' and not a.continue_batch:
            state,code='existing_requires_continue',1
        elif record['state'] in ('duplicate','invalid_choice'):
            state,code='ambiguous_attempts',1
        elif a.continue_batch and record.get('attempts'):
            # Explicit trial-owned attempt only. Failed resume is recorded; next invocation may restart.
            last=record['attempts'][-1]
            path=root/last['run_dir']
            if (path/'checkpoints/restart/state.json').exists() and last.get('state')!='complete':
                cmd+=['--resume',str(path)]
        if a.dry_run:
            print(name, state, subprocess.list2cmdline(cmd),flush=True)
            continue
        started=time.monotonic()
        if state=='pending':
            limit=None
            if a.mode=='full':
                limit=remaining_seconds(a.role,a.billing_start,a.previously_billed_hours,a.hourly_rate)-300
                if limit < projected_seconds(s['env'],s['algo'],observed.get((s['env'],s['algo']),())):
                    state,code='paused_budget',2
            if state=='pending':
                out,err=folder/f'{i:02}.out.log',folder/f'{i:02}.err.log'
                cmd+=['--launcher-stdout',str(out),'--launcher-stderr',str(err)]
                print('START',name,flush=True)
                with out.open('w') as stdout,err.open('w') as stderr:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
                    try:
                        code=proc.wait(timeout=limit)
                    except (subprocess.TimeoutExpired,KeyboardInterrupt):
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill(); proc.wait()
                        code=130
                state='complete' if code==0 and inspect_trial(root,cfg,plan['plan_hash'],choices)['state']=='complete' else 'failed'
                if state=='failed':
                    code=code or 1
                observed.setdefault((s['env'],s['algo']),[]).append(time.monotonic()-started)
        with (folder/'status.csv').open('a',newline='') as f:
            writer=csv.writer(f)
            if f.tell()==0:
                writer.writerow(['trial','state','exit_code','seconds'])
            writer.writerow([name,state,code,time.monotonic()-started])
        print(state.upper(),name,flush=True)
        failures+=bool(code)
        if state=='paused_budget' or code==130:
            print('PAUSED: budget/interrupt. Stop rented machines yourself; this program cannot stop billing.',flush=True)
            break
    print('BATCH FAILED' if failures else 'BATCH COMPLETE',flush=True)
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
