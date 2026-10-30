"""Bundle one machine's stress records: CSV/JSON only, plus a content-addressed manifest.

    python -B run_scripts/stress/collect.py --mode full --role a --out ~/stress_a.zip

Writes <out> (refuses to overwrite) and <out>.sha256 ("<hash>  <name>", sha256sum format).
Excluded: checkpoints, policies, replay, W&B caches and logs. Members keep their paths below
the results root, so bundles from different machines merge without renaming.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.common.stress_io import safe_root  # noqa: E402
from agents.common.stress_plan import RESULTS, digest, load_plan  # noqa: E402
from agents.common.stress_records import manifest  # noqa: E402

# Windows checkout prefix the merged files must fit under (protocol: < 240 characters).
LAPTOP_PREFIX = 92 + len("/run_scripts/stress/results/")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=RESULTS)
    p.add_argument("--mode", choices=("full", "smoke"), default="full")
    p.add_argument("--role", choices=("a", "b", "c"), required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    root = safe_root(a.results_root)
    out = a.out.resolve()
    if out.is_relative_to(root):
        p.error("write the bundle outside the results root")
    plan = load_plan()
    base = root / "v3" / a.mode
    files = sorted(f for f in base.rglob("*") if f.is_file() and f.suffix in (".csv", ".json")
                   and not {"checkpoints", "_wb", "logs", "wandb"} & set(f.relative_to(root).parts))
    for f in files:
        if LAPTOP_PREFIX + len(f.relative_to(root).as_posix()) >= 240:
            raise ValueError(f"nonportable member path: {f}")
    collected = manifest(root, plan, a.mode, role=a.role)
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(root).as_posix())
        z.writestr(f"v3/{a.mode}/collection/{a.role}-{digest(collected)[:12]}.json", json.dumps(collected, indent=1))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out) + ".sha256").write_text(f"{sha}  {out.name}\n", encoding="utf-8", newline="\n")
    done = sum(e["state"] == "complete" for e in collected["entries"])
    print(f"{len(files)} CSV/JSON files from role {a.role} bundled in {out}; {done}/{collected['expected']} runs "
          f"verified complete; sha256 {sha}; no checkpoints or policies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
