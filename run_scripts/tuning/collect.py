"""Create a CSV/JSON-only bundle, excluding checkpoints and W&B caches."""
import argparse
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from agents.common.tuning_plan import RESULTS,load_plan,digest
from agents.common.tuning_io import safe_root,check_path
from agents.common.tuning_records import manifest
import json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,default=RESULTS)
    p.add_argument('--mode',choices=('full','smoke'),default='full')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--attempt-choices',type=Path)
    a=p.parse_args(argv)
    root=safe_root(a.results_root)
    safe_root(a.out.parent)
    plan=load_plan()
    choices=json.loads(a.attempt_choices.read_text()) if a.attempt_choices else None
    files=[f for f in (root/'v3'/a.mode).rglob('*') if f.is_file() and f.suffix in ('.csv','.json')
           and not {'checkpoints','_wb','logs'}.intersection(f.relative_to(root).parts)]
    for f in files:
        # Check the user's 92-character checkout prefix even when packing on Linux.
        if 92+len('/run_scripts/tuning/results/')+len(f.relative_to(root).as_posix())>=240:
            raise ValueError(f'nonportable archive member: {f}')
    # Exclusive output: never clobber earlier collection evidence.
    with zipfile.ZipFile(check_path(a.out),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            name=f.relative_to(root).as_posix()
            if f.name.startswith('billing_') or f.name.startswith('carry_'):
                name=f'accounting/{digest(json.loads(f.read_text()))[:12]}.json'
            z.write(f,name)
        collected=manifest(root,plan,a.mode,choices)
        z.writestr(f'collection/{digest(collected)[:12]}.json',json.dumps(collected,indent=2))
        z.writestr('trials_v3.json',json.dumps(plan,indent=2))
    print(f'{len(files)} CSV/JSON files bundled in {a.out}; no checkpoints')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
