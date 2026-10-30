"""Expected-versus-found stress runs, with optional --role and --level filters.

    python -B run_scripts/stress/check_manifest.py [--mode full] [--role a] [--level hard1] [--json]

Exit 0 only if every selected run is independently verified complete and no unexpected run
directory exists. Missing, failed or unfinished runs are reported as such, never as zeros.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.common.env_factory import STRESS_LEVELS  # noqa: E402
from agents.common.stress_plan import RESULTS, load_plan  # noqa: E402
from agents.common.stress_records import manifest  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=RESULTS)
    p.add_argument("--mode", choices=("full", "smoke"), default="full")
    p.add_argument("--role", choices=("a", "b", "c"))
    p.add_argument("--level", choices=STRESS_LEVELS)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    m = manifest(a.results_root, load_plan(), a.mode, a.role, a.level)
    if a.json:
        print(json.dumps(m, indent=2))
    else:
        for e in m["entries"]:
            if e["state"] != "complete":
                problems = "; ".join(p for at in e.get("attempts", []) for p in at.get("problems", []))[:300]
                print(f"{e['state']:<12} role {e['role']} slot {e['slot']:>3} {e['env']}/{e['algo']}/{e['level']}/s{e['seed']}"
                      + (f"  ({problems})" if problems else ""))
        for u in m["unexpected_runs"]:
            print(f"unexpected   {u}")
        done = sum(e["state"] == "complete" for e in m["entries"])
        scope = ", ".join(f"{k} {v}" for k, v in m["filters"].items() if v) or "all runs"
        print(f"{done}/{m['expected']} complete ({scope}, mode {a.mode}); by role {m['by_role']}; "
              f"unexpected {len(m['unexpected_runs'])}")
    return 0 if m["all_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
