"""Expected-versus-collected tuning slots; JSON includes counts for each role."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.common.tuning_plan import RESULTS,load_plan
from agents.common.tuning_records import manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,default=RESULTS)
    p.add_argument('--mode',choices=('full','smoke'),default='full')
    p.add_argument('--json',action='store_true')
    p.add_argument('--role',choices=('a','b','c'))
    p.add_argument('--stage',choices=('screen','confirm'))
    p.add_argument('--attempt-choices',type=Path)
    a=p.parse_args(argv)
    choices=json.loads(a.attempt_choices.read_text()) if a.attempt_choices else None
    r=manifest(a.results_root.resolve(),load_plan(),a.mode,choices)
    subset=[s for s in r['entries'] if (not a.role or s['role']==a.role) and (not a.stage or s['stage']==a.stage)]
    ok=all(s['state']=='complete' for s in subset) and not r['unexpected_runs']
    print(json.dumps(r,indent=2) if a.json else f"{sum(s['state']=='complete' for s in subset)}/{len(subset)} complete; roles: {r['by_role']}")
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
