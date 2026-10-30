"""Pack the final policy of every verified stress run of one role, or verify such an archive.

    python -B run_scripts/stress/pack_models.py pack --mode full --role a --out ~/stress_models_a.tar.gz
    python -B run_scripts/stress/pack_models.py verify --archive <path>.tar.gz

pack: only runs that pass independent verification are included, one final policy each
(checkpoints/policy_step<budget>.zip or .pt). The archive must be written outside this
checkout. It contains models_manifest.json (run, config hash, plan hash, file SHA-256) and is
accompanied by <archive>.sha256 ("<hash>  <name>"). Refuses to overwrite.
verify: checks <archive>.sha256, then every member against models_manifest.json.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path_or_bytes):
    h = hashlib.sha256()
    if isinstance(path_or_bytes, bytes):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def pack(a):
    from agents.common.stress_io import safe_root
    from agents.common.stress_plan import RESULTS, load_plan
    from agents.common.stress_records import manifest
    out = Path(a.out).expanduser().resolve()
    if out.is_relative_to(ROOT.resolve()):
        raise SystemExit("refusing to write models inside the checkout; choose a path outside it")
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    root = safe_root(a.results_root or RESULTS)
    plan = load_plan()
    m = manifest(root, plan, a.mode, role=a.role)
    entries, missing = [], []
    for e in m["entries"]:
        if e["state"] != "complete" or len(e.get("final_policy", [])) != 1:
            missing.append(f"{e['env']}/{e['algo']}/{e['level']}/s{e['seed']} ({e['state']})")
            continue
        src = root / e["run_dir"] / "checkpoints" / e["final_policy"][0]
        member = f"{e['run_dir']}/{e['final_policy'][0]}"
        entries.append({"member": member, "sha256": sha256(src), "bytes": src.stat().st_size,
                        "env": e["env"], "algo": e["algo"], "level": e["level"], "seed": e["seed"],
                        "config_hash": e["config_hash"], "src": src})
    if missing and not a.allow_missing:
        raise SystemExit("runs without a verified final policy (use --allow-missing to pack the rest):\n  "
                         + "\n  ".join(missing))
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"phase": "stress", "mode": a.mode, "role": a.role, "plan_hash": plan["plan_hash"],
            "files": [{k: v for k, v in x.items() if k != "src"} for x in entries], "missing": missing}
    with tarfile.open(out, "x:gz") as tar:
        for x in entries:
            tar.add(x["src"], arcname=x["member"])
        blob = json.dumps(meta, indent=1).encode("utf-8")
        info = tarfile.TarInfo("models_manifest.json")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    digest = sha256(out)
    Path(str(out) + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8", newline="\n")
    size = sum(x["bytes"] for x in entries) / 1e6
    print(f"MODELS PACKED: {len(entries)} final policies ({size:.1f} MB) for role {a.role} in {out}; "
          f"sha256 {digest}" + (f"; {len(missing)} missing" if missing else ""))
    return 0


def verify(a):
    archive = Path(a.archive).expanduser().resolve()
    expected = Path(str(archive) + ".sha256").read_text(encoding="utf-8").split()[0].lower()
    if sha256(archive) != expected:
        raise SystemExit(f"HASH MISMATCH: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        meta = json.loads(tar.extractfile("models_manifest.json").read())
        names = {m.name for m in tar.getmembers() if m.isfile()} - {"models_manifest.json"}
        if names != {f["member"] for f in meta["files"]}:
            raise SystemExit("archive members differ from models_manifest.json")
        for f in meta["files"]:
            if sha256(tar.extractfile(f["member"]).read()) != f["sha256"]:
                raise SystemExit(f"member hash mismatch: {f['member']}")
    print(f"MODELS VERIFIED: {len(meta['files'])} final policies, role {meta['role']}, mode {meta['mode']}, "
          f"plan {meta['plan_hash'][:12]}" + (f", {len(meta['missing'])} missing" if meta["missing"] else ""))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pack")
    pk.add_argument("--results-root", type=Path)
    pk.add_argument("--mode", choices=("full", "smoke"), default="full")
    pk.add_argument("--role", choices=("a", "b", "c"), required=True)
    pk.add_argument("--out", required=True)
    pk.add_argument("--allow-missing", action="store_true")
    vf = sub.add_parser("verify")
    vf.add_argument("--archive", required=True)
    a = p.parse_args(argv)
    return pack(a) if a.cmd == "pack" else verify(a)


if __name__ == "__main__":
    raise SystemExit(main())
