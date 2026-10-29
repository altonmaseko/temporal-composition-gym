"""Persist conservative rental accounting across stopped instances (no cloud API)."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from agents.common.tuning_plan import RESULTS
from agents.common.tuning_io import safe_root
from agents.common.run_io import write_json_atomic

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',required=True,choices=list('abc'))
    p.add_argument('--start')
    p.add_argument('--close',action='store_true')
    a=p.parse_args()
    root=safe_root(RESULTS)/'v3/full'; root.mkdir(parents=True,exist_ok=True)
    ledger=root/f'billing_{a.role}.json'; carry=root/f'carry_{a.role}.json'
    env=Path.home()/f'tuning_{a.role}.env'
    if a.close:
        terms=json.loads(ledger.read_text())
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(terms['billing_start'].replace('Z','+00:00'))).total_seconds()/3600
        hours=terms['previously_billed_hours']+elapsed+1
        write_json_atomic(str(carry),{'hours':hours,'stop_allowance_hours':1})
        ledger.rename(root/f"bill_{a.role}_{datetime.now(timezone.utc):%m%d%H%M%S}.json")
        print(f'Stop the instance in the console within one hour. Carried hours: {hours:.6f}')
    else:
        start=datetime.fromisoformat(a.start.replace('Z','+00:00'))
        if start.tzinfo is None or start>datetime.now(timezone.utc):
            p.error('supply the actual past UTC rental start with Z')
        if ledger.exists():
            p.error('session already active; source the existing env file, or close it before a new session')
        previous=json.loads(carry.read_text())['hours'] if carry.exists() else 0
        env.write_text(f"BILLING_START='{start.isoformat()}'\nPREVIOUS_HOURS='{previous:.9f}'\n",encoding='utf-8')
        print(f'Session ready: {previous:.6f} previous hours, rate $0.30/h')

if __name__=='__main__': main()
