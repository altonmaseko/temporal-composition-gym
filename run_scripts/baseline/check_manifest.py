"""Expected-versus-collected manifest for the protocol-v2 baseline matrix.

Scans one or more results roots (e.g. the three machines' copies, or one merged tree) and
classifies every planned (env, algo, seed) run:

  complete      exactly one verified run with the planned config hash
  duplicate     more than one verified run for the same planned run (choose explicitly)
  nonstandard   verified run(s) whose config hash differs from the plan (overrides) -- not counted
  failed        only failed / interrupted / still-running attempts
  missing       nothing found

A run is "verified" only if status.json says complete+verified AND the files agree: budget
reached, all periodic evaluation points, all final-evaluation episodes, final_summary.json.
Missing or failed runs are reported as such; nothing is treated as a zero result.

    python run_scripts/baseline/check_manifest.py                       # default results root
    python run_scripts/baseline/check_manifest.py --root A --root B --json manifest.json
Exit code 0 only if every planned run is complete exactly once.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.common.baseline_plan import expected_runs  # noqa: E402
from agents.common.env_factory import PROTOCOL_VERSION  # noqa: E402
from agents.common.run_io import DEFAULT_RESULTS_ROOT, read_json  # noqa: E402
from agents.common.seeding import BASELINE_TRAIN_SEEDS  # noqa: E402


def _rows(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_run(run_dir):
    """(verified: bool, problems: list[str], info: dict) for one run directory."""
    problems = []
    try:
        identity = read_json(os.path.join(run_dir, "identity.json"))
        cfg = read_json(os.path.join(run_dir, "config.resolved.json"))
    except (OSError, ValueError) as exc:
        return False, [f"unreadable identity/config: {exc}"], {}
    status = read_json(os.path.join(run_dir, "status.json")) if os.path.exists(os.path.join(run_dir, "status.json")) else {}
    info = {"run_dir": run_dir, "run_id": identity.get("run_id"), "config_hash": identity.get("config_hash"),
            "state": status.get("state", "unknown"), "env": identity.get("env"), "algo": identity.get("algo"),
            "seed": identity.get("seed"), "mode": identity.get("mode")}
    if status.get("state") != "complete" or not status.get("verified"):
        problems.append(f"status is {status.get('state')!r}" + (f": {status.get('error')}" if status.get("error") else ""))
        return False, problems, info
    budget = cfg["budget"]["env_steps"]
    if status.get("env_steps_done") != budget:
        problems.append(f"env_steps_done {status.get('env_steps_done')} != budget {budget}")
    points = _rows(os.path.join(run_dir, "eval_points.csv"))
    if points is None or len(points) != cfg["eval"]["n_points"] + 1:
        problems.append(f"eval points {None if points is None else len(points)} != {cfg['eval']['n_points'] + 1}")
    elif int(points[-1]["global_step"]) != budget:
        problems.append("last periodic evaluation is not at the budget")
    final = _rows(os.path.join(run_dir, "eval_final.csv"))
    if final is None or len(final) != cfg["eval"]["final_episodes"]:
        problems.append(f"final eval episodes {None if final is None else len(final)} != {cfg['eval']['final_episodes']}")
    if not os.path.exists(os.path.join(run_dir, "final_summary.json")):
        problems.append("final_summary.json missing")
    return not problems, problems, info


def scan(roots, mode):
    found = defaultdict(list)
    for root in roots:
        pattern = os.path.join(root, PROTOCOL_VERSION, mode, "*", "*", "seed*__*", "identity.json")
        for ident in glob.glob(pattern):
            run_dir = os.path.dirname(ident)
            ok, problems, info = verify_run(run_dir)
            info.update(verified=ok, problems=problems)
            found[(info.get("env"), info.get("algo"), info.get("seed"))].append(info)
    return found


def build_manifest(roots, mode="full", seeds=BASELINE_TRAIN_SEEDS):
    found = scan(roots, mode)
    entries = []
    for exp in expected_runs(mode, seeds):
        key = (exp["env"], exp["algo"], exp["seed"])
        runs = found.pop(key, [])
        good = [r for r in runs if r["verified"] and r["config_hash"] == exp["config_hash"]]
        odd = [r for r in runs if r["verified"] and r["config_hash"] != exp["config_hash"]]
        if len(good) == 1:
            state = "complete"
        elif len(good) > 1:
            state = "duplicate"
        elif odd:
            state = "nonstandard"
        elif runs:
            state = "failed"
        else:
            state = "missing"
        entries.append({**exp, "state": state, "attempts": runs})
    unexpected = [r for runs in found.values() for r in runs]
    counts = defaultdict(int)
    for e in entries:
        counts[e["state"]] += 1
    return {"protocol_version": PROTOCOL_VERSION, "mode": mode, "roots": [os.path.abspath(r) for r in roots],
            "expected": len(entries), "counts": dict(counts), "entries": entries, "unexpected_runs": unexpected,
            "all_complete": counts["complete"] == len(entries)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", action="append", default=None, help=f"results root (repeatable; default {DEFAULT_RESULTS_ROOT})")
    p.add_argument("--mode", choices=("full", "smoke"), default="full")
    p.add_argument("--seeds", type=int, nargs="+", default=list(BASELINE_TRAIN_SEEDS))
    p.add_argument("--json", default=None, help="write the manifest as JSON")
    args = p.parse_args(argv)
    m = build_manifest(args.root or [DEFAULT_RESULTS_ROOT], args.mode, args.seeds)
    for e in m["entries"]:
        detail = ""
        if e["state"] != "complete":
            detail = "; ".join(sorted({pr for a in e["attempts"] for pr in a["problems"]}))[:120]
        print(f"{e['state']:<12} {e['role']:<11} {e['env']:<12} {e['algo']:<14} seed{e['seed']}  {detail}")
    for r in m["unexpected_runs"]:
        print(f"{'unexpected':<12} {r['run_dir']}")
    print(f"\n{m['counts'].get('complete', 0)}/{m['expected']} planned runs complete; counts: {m['counts']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
    return 0 if m["all_complete"] and not m["unexpected_runs"] else 1


if __name__ == "__main__":
    sys.exit(main())
