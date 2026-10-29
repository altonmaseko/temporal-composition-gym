"""Freeze top-two assignments, or select using all confirmation evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.common.tuning_plan import RESULTS, load_plan
from agents.common.tuning_records import freeze_finalists, select


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-confirmation', action='store_true')
    p.add_argument('--mode', choices=('full','smoke'), default='full')
    p.add_argument('--results-root', type=Path, default=RESULTS)
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--reason')
    p.add_argument('--attempt-choices', type=Path, help='JSON mapping config hash to relative attempt directory')
    a = p.parse_args(argv)
    choices = json.loads(a.attempt_choices.read_text()) if a.attempt_choices else None
    try:
        result = (freeze_finalists if a.prepare_confirmation else select)(a.results_root,load_plan(),a.mode,a.allow_incomplete,a.reason,choices)
    except ValueError as exc:
        print(str(exc),file=sys.stderr)
        return 1
    print('Finalists frozen' if a.prepare_confirmation else json.dumps({'complete':result['complete'],'problems':result['problems']}))
    return 0 if a.prepare_confirmation or result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
