"""Stress execution and collection pipeline: launcher banners and guards, verification,
collection, model packing, manifest filters and the preflight plan check."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

import pytest

from agents.common.stress_plan import ROOT, load_plan, run_config
from agents.common.stress_records import inspect_run, manifest, prune_completed

GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
CELL = dict(env="delivery", algo="dqn", level="default", seed=301)


@pytest.fixture(scope="module")
def plan():
    return load_plan()


@pytest.fixture(scope="module")
def smoke_root(tmp_path_factory):
    """One real smoke run (delivery DQN, default level, seed 301) shared by several tests."""
    from run_scripts.stress.run_stress import main
    root = tmp_path_factory.mktemp("sr")
    code = main(["--env", "delivery", "--algo", "dqn", "--level", "default", "--seed", "301", "--mode", "smoke",
                 "--device", "cpu", "--threads", "1", "--wandb-mode", "disabled", "--results-root", str(root)])
    assert code == 0
    return root


def _entry_cfg(plan, **cell):
    return run_config(plan, cell["env"], cell["algo"], cell["level"], cell["seed"], "smoke")


# --------------------------------------------------------------------------- verification
def test_verify_and_prune_a_real_smoke_run(plan, smoke_root):
    entry, cfg = _entry_cfg(plan, **CELL)
    rec = inspect_run(smoke_root, cfg, plan["plan_hash"], entry)
    assert rec["state"] == "complete", rec
    assert rec["final_policy"] == ["policy_step000000064.zip"]
    run = smoke_root / rec["run_dir"]
    assert (run / "checkpoints/restart").exists()
    assert prune_completed(smoke_root, cfg, plan["plan_hash"], entry)
    assert not (run / "checkpoints/restart").exists() and (run / "checkpoints/policy_step000000064.zip").exists()
    m = manifest(smoke_root, plan, "smoke", role=entry["role"], level="default")
    assert m["counts"]["complete"] == 1 and m["by_level"]["default"]["complete"] == 1
    assert {e["level"] for e in m["entries"]} == {"default"} and {e["role"] for e in m["entries"]} == {entry["role"]}
    assert not m["unexpected_runs"] and not m["all_complete"]          # other runs of the role are missing


@pytest.mark.parametrize("damage", ["seed", "required", "identity", "summary"])
def test_verification_rejects_damaged_records(plan, smoke_root, tmp_path, damage):
    import shutil
    entry, cfg = _entry_cfg(plan, **CELL)
    rec = inspect_run(smoke_root, cfg, plan["plan_hash"], entry)
    root = tmp_path / "copy"
    shutil.copytree(smoke_root / "v3", root / "v3")
    run = root / rec["run_dir"]
    if damage in ("seed", "required"):
        path = run / "eval_final.csv"
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        if damage == "seed":
            rows[0]["episode_seed"] = "4010000"            # stress block instead of the final-test block
        else:
            rows[0]["required"] = ""
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    elif damage == "identity":
        ident = json.loads((run / "identity.json").read_text())
        ident["level"] = "easy"
        (run / "identity.json").write_text(json.dumps(ident))
    else:
        summ = json.loads((run / "final_summary.json").read_text())
        summ["n_updates"] += 1
        (run / "final_summary.json").write_text(json.dumps(summ))
    bad = inspect_run(root, cfg, plan["plan_hash"], entry)
    assert bad["state"] == "failed" and bad["attempts"][0]["problems"]


# --------------------------------------------------------------------------- collection and packing
def test_collect_bundles_csv_json_only_with_hash(plan, smoke_root, tmp_path):
    from run_scripts.stress.collect import main
    out = tmp_path / "stress_a.zip"
    role = run_config(plan, **CELL, mode="smoke")[0]["role"]
    assert main(["--results-root", str(smoke_root), "--mode", "smoke", "--role", role, "--out", str(out)]) == 0
    names = zipfile.ZipFile(out).namelist()
    assert names and all(n.endswith((".csv", ".json")) for n in names)
    assert not any("checkpoints" in n or "_wb" in n for n in names)
    assert any(n.startswith(f"v3/smoke/collection/{role}-") for n in names)
    digest, name = (tmp_path / "stress_a.zip.sha256").read_text().split()
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest() and name == out.name
    with pytest.raises(FileExistsError):
        main(["--results-root", str(smoke_root), "--mode", "smoke", "--role", role, "--out", str(out)])
    with pytest.raises(SystemExit):
        main(["--results-root", str(smoke_root), "--mode", "smoke", "--role", role, "--out", str(smoke_root / "x.zip")])


def test_pack_and_verify_models(plan, smoke_root, tmp_path, capsys):
    from run_scripts.stress.pack_models import main
    role = run_config(plan, **CELL, mode="smoke")[0]["role"]
    out = tmp_path / "models.tar.gz"
    with pytest.raises(SystemExit):                       # other runs of the role are missing
        main(["pack", "--results-root", str(smoke_root), "--mode", "smoke", "--role", role, "--out", str(out)])
    assert main(["pack", "--results-root", str(smoke_root), "--mode", "smoke", "--role", role, "--out", str(out),
                 "--allow-missing"]) == 0
    assert main(["verify", "--archive", str(out)]) == 0
    with tarfile.open(out) as tar:
        members = [m.name for m in tar.getmembers()]
    assert "models_manifest.json" in members and any(m.endswith("policy_step000000064.zip") for m in members)
    (tmp_path / "models.tar.gz.sha256").write_text("0" * 64 + "  models.tar.gz\n")
    with pytest.raises(SystemExit, match="HASH MISMATCH"):
        main(["verify", "--archive", str(out)])
    with pytest.raises(SystemExit, match="inside the checkout"):
        main(["pack", "--results-root", str(smoke_root), "--mode", "smoke", "--role", role,
              "--out", str(ROOT / "run_scripts/stress/results/x.tar.gz"), "--allow-missing"])


# --------------------------------------------------------------------------- launcher
def _one_job_plan(plan, role="b"):
    small = copy.deepcopy(plan)
    small["runs"] = [r for r in plan["runs"] if r["role"] == role][:2]
    return small


def test_launcher_guards_and_dry_run(tmp_path, capsys):
    from run_scripts.stress import launch
    with pytest.raises(SystemExit):
        launch.main(["--role", "a", "--threads", "4", "--dry-run"])
    with pytest.raises(SystemExit):
        launch.main(["--role", "a", "--threads", "1", "--device-policy", "cpu", "--dry-run"])
    with pytest.raises(SystemExit):
        launch.main(["--role", "a", "--threads", "1", "--debug-fail-run", "x@1", "--dry-run"])
    with pytest.raises(SystemExit):
        launch.main(["--role", "a"])                           # --threads is required
    root = tmp_path / "r"
    assert launch.main(["--role", "c", "--mode", "smoke", "--threads", "1", "--device-policy", "cpu", "--dry-run",
                        "--results-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "role c: 46 runs" in out and "ESTIMATE at launch: 50.3 h" in out and "--threads 1" in out
    assert not root.exists()


def test_launcher_failure_pause_and_continue(tmp_path, monkeypatch, capsys, plan):
    from run_scripts.stress import launch
    monkeypatch.setattr(launch, "load_plan", lambda: _one_job_plan(plan))
    monkeypatch.setattr(launch, "prune_completed", lambda *a: True)
    args = ["--role", "b", "--mode", "smoke", "--threads", "1", "--device-policy", "cpu", "--wandb-mode", "disabled"]

    class Proc:
        code = 7
        def __init__(self, cmd, **kw):
            Proc.cmd = cmd
        def wait(self, **kw):
            if Proc.code == "interrupt":
                if "timeout" in kw:                        # after terminate(): the child has exited
                    return -15
                raise KeyboardInterrupt
            return Proc.code
        def terminate(self):
            pass
    monkeypatch.setattr(launch.subprocess, "Popen", Proc)
    assert launch.main(args + ["--results-root", str(tmp_path / "r1")]) == 1
    out = capsys.readouterr().out
    assert "BATCH FAILED" in out and "FAILED delivery/dqn/default/s302" in out and "ESTIMATE:" in out
    status = next((tmp_path / "r1").rglob("status.csv")).read_text()
    assert ",failed,7," in status
    progress = json.loads(next((tmp_path / "r1").rglob("progress.json")).read_text())
    assert progress["done"] == 2 and progress["failures"] == 2
    Proc.code = "interrupt"
    assert launch.main(args + ["--results-root", str(tmp_path / "r2")]) == 130
    assert "PAUSED" in capsys.readouterr().out
    monkeypatch.setattr(launch, "inspect_run", lambda *a: {"state": "complete", "attempts": [{}]})
    assert launch.main(args + ["--results-root", str(tmp_path / "r3"), "--continue-batch"]) == 0
    out = capsys.readouterr().out
    assert out.count("SKIPPED_COMPLETE") == 2 and "BATCH COMPLETE" in out
    assert launch.main(args + ["--results-root", str(tmp_path / "r4")]) == 1   # existing runs need --continue-batch
    assert "EXISTING_REQUIRES_CONTINUE" in capsys.readouterr().out


def test_injected_failure_flag_reaches_only_the_named_run(tmp_path, monkeypatch, plan):
    from run_scripts.stress import launch
    monkeypatch.setattr(launch, "load_plan", lambda: _one_job_plan(plan))
    cmds = []

    class Proc:
        def __init__(self, cmd, **kw):
            cmds.append(cmd)
        def wait(self, **kw):
            return 0
    monkeypatch.setattr(launch.subprocess, "Popen", Proc)
    monkeypatch.setattr(launch, "prune_completed", lambda *a: True)
    launch.main(["--role", "b", "--mode", "smoke", "--threads", "1", "--device-policy", "cpu", "--wandb-mode",
                 "disabled", "--results-root", str(tmp_path), "--debug-fail-run", "delivery/dqn_ts/default/s302@40"])
    flagged = [c for c in cmds if "--debug-fail-at-step" in c]
    assert len(cmds) == 2 and len(flagged) == 1 and "dqn_ts" in flagged[0]


def test_estimate_rescales_by_observed_speed(plan):
    from run_scripts.stress.launch import Estimate
    jobs = [r for r in plan["runs"] if r["role"] == "a"]
    est = Estimate(jobs)
    assert est.remaining(jobs) == pytest.approx(50.2, abs=0.1)
    est.record(jobs[0], 2 * 0.18 * 3600)                    # twice as slow as planned
    assert est.factor() == pytest.approx(2.0)
    assert est.remaining(jobs[1:]) == pytest.approx(2 * (50.2 - 0.18), abs=0.2)


# --------------------------------------------------------------------------- preflight and scripts
def test_preflight_plan_check_detects_a_changed_plan(tmp_path, monkeypatch):
    from agents.common import stress_plan
    from run_scripts.baseline.preflight import Report
    from run_scripts.stress.preflight import check_plan
    rep = Report()
    assert check_plan(rep, 1, "full") is not None and not rep.failed
    bad = json.loads(stress_plan.PLAN_PATH.read_text())
    bad["runs"][0]["config"]["algo"]["hyperparams"]["learning_rate"] = 0.00050000000000001
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(bad))
    monkeypatch.setattr(stress_plan, "PLAN_PATH", path)
    rep = Report()
    assert check_plan(rep, 1, "full") is None and rep.failed
    assert "learning_rate" in rep.items[0]["detail"]
    rep = Report()
    monkeypatch.undo()
    check_plan(rep, 4, "full")
    assert rep.failed                                      # --threads must match the plan in full mode


@pytest.mark.skipif(not GIT_BASH.exists(), reason="Git Bash not installed")
def test_shell_wrappers_parse_and_require_python_and_threads():
    d = ROOT / "run_scripts/stress"
    for f in ("lib_launcher.sh", "run_a_stress.sh", "run_b_stress.sh", "run_c_stress.sh"):
        assert subprocess.run([str(GIT_BASH), "-n", str(d / f)]).returncode == 0
        assert b"\r" not in (d / f).read_bytes()
    script = (d / "run_a_stress.sh").as_posix()
    r = subprocess.run([str(GIT_BASH), script, "--threads", "1"], capture_output=True, text=True)
    assert r.returncode == 2 and "--python" in r.stderr
    r = subprocess.run([str(GIT_BASH), script, "--python", "/usr/bin/python3"], capture_output=True, text=True)
    assert r.returncode == 2 and "--threads" in r.stderr
