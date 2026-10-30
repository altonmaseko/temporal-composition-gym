"""Write or check the frozen stress plan (run_scripts/stress/stress_plan_v3.json).

    python -B run_scripts/stress/generate_plan.py           # write (refuses to replace a different plan)
    python -B run_scripts/stress/generate_plan.py --check   # rebuild and require exact JSON equality

The plan is rebuilt from STRESS_PROTOCOL.md's frozen levels and the published tuning
selection, with no random sampling. --check prints every differing field on mismatch.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.common.stress_plan import PLAN_PATH, build_plan, json_diff  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--out", type=Path, default=PLAN_PATH)
    args = p.parse_args(argv)
    plan = build_plan()
    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist", file=sys.stderr)
            return 1
        on_disk = json.loads(args.out.read_text(encoding="utf-8"))
        if on_disk != plan:
            diff = json_diff(on_disk, plan)
            print(f"{args.out.name} differs from the frozen generator ({len(diff)} field(s) shown):", file=sys.stderr)
            for line in diff:
                print("  " + line, file=sys.stderr)
            return 1
        print(f"{args.out.name} matches the frozen generator: {plan['run_count']} runs, "
              f"roles {plan['role_counts']}, plan hash {plan['plan_hash']}")
        return 0
    if args.out.exists() and json.loads(args.out.read_text(encoding="utf-8")) != plan:
        print("refusing to overwrite a different stress plan; a change needs a new dated decision", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(plan, indent=1) + "\n")
    print(f"wrote {args.out} ({plan['run_count']} runs, plan hash {plan['plan_hash']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
