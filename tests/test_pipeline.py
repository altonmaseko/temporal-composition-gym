"""Baseline pipeline (Prompt 3): plan, run identity, records, evaluation isolation, resume, launchers."""
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from agents.common import baseline_plan as plan
from agents.common.run_io import RECORDED_ENV_VARS, read_json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = dict(os.environ, SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
# A cheap delivery DQN configuration for pipeline mechanics (smoke mode + explicit overrides).
TINY = ["--set", "budget.env_steps=800", "--set", "eval.every_steps=400", "--set", "checkpoint.every_steps=400",
        "--set", "algo.hyperparams.learning_starts=200", "--set", "logging.train_log_every_steps=200"]


def run_script(script, *args, root):
    cmd = [PY, os.path.join(REPO, script), *args, "--results-root", str(root)]
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV, cwd=str(root))


def run_dirs(root, mode="smoke"):
    base = os.path.join(root, "v2", mode)
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        if "identity.json" in filenames:
            out.append(dirpath)
    return sorted(out)


def csv_rows(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip().splitlines()[1:]


# --------------------------------------------------------------------------- plan
def test_plan_covers_the_agreed_matrix():
    runs = plan.expected_runs()
    assert len(runs) == 55 and sum(r["family"] == "control" for r in runs) == 10
    cells = {(r["env"], r["algo"]) for r in runs}
    assert cells == {("delivery", "dqn"), ("colour", "dqn"), ("maintenance", "ppo"),
                     ("delivery", "drqn"), ("colour", "drqn"), ("maintenance", "recurrent_ppo"),
                     ("delivery", "qrm"), ("colour", "qrm_cue"), ("maintenance", "ppo_rm"),
                     ("delivery", "dqn_ts"), ("colour", "dqn_ts")}
    by_role = {role: [r for r in runs if r["role"] == role] for role in plan.ROLES}
    assert {k: len(v) for k, v in by_role.items()} == {"memoryless": 25, "implicit": 15, "explicit": 15}
    # seed-major: all three main cells of seed 1 come before any seed-2 run; controls last
    order = plan.role_runs("memoryless")
    assert [s for _, _, s in order[:3]] == [1, 1, 1] and all(a == "dqn_ts" for _, a, _ in order[15:])
    for r in runs:                                   # identical budgets within an environment
        assert plan.resolve_config(r["env"], r["algo"], r["seed"])["budget"]["env_steps"] == plan.BUDGETS[r["env"]]
    assert plan.main(["write", "--check"]) == 0      # baseline_plan.json is current


def test_config_hash_overrides_and_guards():
    a = plan.resolve_config("delivery", "dqn", 1)
    assert plan.config_hash(a) == plan.config_hash(plan.resolve_config("delivery", "dqn", 1))
    assert plan.config_hash(a) != plan.config_hash(plan.resolve_config("delivery", "dqn", 2))
    assert plan.config_hash(a) != plan.config_hash(plan.resolve_config("delivery", "dqn", 1, mode="smoke"))
    b = plan.resolve_config("delivery", "dqn", 1, overrides={"algo.hyperparams.learning_rate": 1e-3})
    assert b["overrides"] == {"algo.hyperparams.learning_rate": 1e-3} and plan.config_hash(b) != plan.config_hash(a)
    # the W&B destination is recorded but does not change the experiment's identity
    c = plan.resolve_config("delivery", "dqn", 1, wandb_entity="someone")
    assert plan.config_hash(c) == plan.config_hash(a) and c["wandb"]["entity"] == "someone"
    assert a["wandb"]["project"] == "new_experiments"
    with pytest.raises(KeyError):
        plan.resolve_config("delivery", "dqn", 1, overrides={"algo.hyperparams.not_a_setting": 1})
    with pytest.raises(KeyError):
        plan.resolve_config("delivery", "dqn", 1, overrides={"seed.train_seed": 3})
    with pytest.raises(ValueError):
        plan.resolve_config("delivery", "dqn", 1, wandb_project="temporal-composition-gym")
    with pytest.raises(ValueError):
        plan.resolve_config("delivery", "dqn", 101)                    # tuning seed range
    with pytest.raises(ValueError):
        plan.resolve_config("maintenance", "ppo", 1, overrides={"budget.env_steps": 2_048_100})   # SB3 rounding


# --------------------------------------------------------------------------- runs
def test_smoke_run_writes_complete_isolated_records(tmp_path):
    res = run_script("agents/train_dqn.py", "--env", "delivery", "--seed", "1", "--mode", "smoke",
                     "--wandb-mode", "offline", *TINY, root=tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    (rd,) = run_dirs(tmp_path)
    assert os.sep.join(["v2", "smoke", "delivery", "dqn"]) in rd
    status, ident = read_json(os.path.join(rd, "status.json")), read_json(os.path.join(rd, "identity.json"))
    cfg = read_json(os.path.join(rd, "config.resolved.json"))
    assert status["state"] == "complete" and status["verified"] and status["env_steps_done"] == 800
    assert ident["config_hash"] == plan.config_hash(cfg) and ident["mode"] == "smoke"
    assert cfg["wandb"]["group"].endswith("-smoke") and cfg["overrides"]
    assert len(csv_rows(os.path.join(rd, "eval_points.csv"))) == 3
    assert len(csv_rows(os.path.join(rd, "eval_final.csv"))) == cfg["eval"]["final_episodes"]
    assert len(csv_rows(os.path.join(rd, "train_updates.csv"))) >= 2
    meta = read_json(os.path.join(rd, "metadata.json"))
    assert meta["git"]["rev"] and meta["hardware"]["cpu_count_logical"] and meta["device"]["selected"] == "cpu"
    assert set(meta["env_vars"]) == set(RECORDED_ENV_VARS)
    for name in ("metadata.json", "config.resolved.json", "identity.json", "status.json"):
        text = open(os.path.join(rd, name), encoding="utf-8").read().lower()
        assert "api_key" not in text and "wandb_api" not in text
    assert any(d.startswith("offline-run") for d in os.listdir(os.path.join(rd, "wandb")))
    restart = os.path.join(rd, "checkpoints", "restart")
    assert {"model.zip", "replay.pkl", "state.pt", "state.json"} <= set(os.listdir(restart))


def test_fresh_runs_never_collide_or_inherit_state(tmp_path):
    for seed in ("1", "1", "2"):
        res = run_script("agents/train_dqn.py", "--env", "delivery", "--seed", seed, "--mode", "smoke",
                         "--wandb-mode", "disabled", *TINY, root=tmp_path)
        assert res.returncode == 0, res.stderr[-2000:]
    other = run_script("agents/train_dqn.py", "--env", "delivery", "--seed", "1", "--mode", "smoke",
                       "--wandb-mode", "disabled", *TINY, "--set", "algo.hyperparams.learning_rate=0.001", root=tmp_path)
    assert other.returncode == 0
    dirs = run_dirs(tmp_path)
    assert len(dirs) == 4 and len(set(dirs)) == 4
    idents = [read_json(os.path.join(d, "identity.json")) for d in dirs]
    assert len({i["run_id"] for i in idents}) == 4 and len({i["wandb_id"] for i in idents}) == 4
    assert len({i["config_hash"] for i in idents}) == 3                # seed 1 twice share the hash
    first_eval = {}
    for d, i in zip(dirs, idents):
        st = read_json(os.path.join(d, "status.json"))
        assert st["restored_steps"] == 0 and st["new_steps_this_session"] == 800
        row0 = csv_rows(os.path.join(d, "eval_points.csv"))[0]
        first_eval.setdefault((i["config_hash"]), set()).add(row0)
    # same config + seed -> identical untrained policy at step 0 (nothing inherited from earlier runs)
    assert all(len(v) == 1 for v in first_eval.values())


def test_injected_failure_is_recorded_as_failed_not_zero(tmp_path):
    res = run_script("agents/train_dqn.py", "--env", "delivery", "--seed", "1", "--mode", "smoke",
                     "--wandb-mode", "disabled", "--debug-fail-at-step", "500", root=tmp_path)
    assert res.returncode == 1 and "injected failure" in res.stderr
    (rd,) = run_dirs(tmp_path)
    st = read_json(os.path.join(rd, "status.json"))
    assert st["state"] == "failed" and "injected failure" in st["error"] and not st.get("verified")
    assert not os.path.exists(os.path.join(rd, "final_summary.json"))
    sys.path.insert(0, os.path.join(REPO, "run_scripts", "baseline"))
    import check_manifest
    m = check_manifest.build_manifest([str(tmp_path)], mode="smoke", seeds=[1])
    entry = next(e for e in m["entries"] if e["env"] == "delivery" and e["algo"] == "dqn")
    assert entry["state"] == "failed" and not m["all_complete"]
    assert all(e["state"] == "missing" for e in m["entries"] if e is not entry)


@pytest.mark.slow
@pytest.mark.parametrize("script,env,expected_updates", [
    ("agents/drqn/train_drqn.py", "delivery", (4000 - 500) // 4 + 1),
    ("agents/train_dqn.py", "delivery", (4000 - 500) // 4),
])
def test_resume_restores_the_run(tmp_path, script, env, expected_updates):
    args = ["--env", env, "--seed", "2", "--mode", "smoke", "--wandb-mode", "disabled"]
    first = run_script(script, *args, "--debug-fail-at-step", "3000", root=tmp_path)
    assert first.returncode == 1
    (rd,) = run_dirs(tmp_path)
    assert read_json(os.path.join(rd, "checkpoints", "restart", "state.json"))["global_step"] == 2000
    wrong = run_script(script, "--env", env, "--seed", "3", "--mode", "smoke", "--wandb-mode", "disabled",
                       "--resume", rd, root=tmp_path)
    assert wrong.returncode != 0 and "resume refused" in wrong.stderr
    second = run_script(script, *args, "--resume", rd, root=tmp_path)
    assert second.returncode == 0, second.stderr[-2000:]
    st = read_json(os.path.join(rd, "status.json"))
    assert st["state"] == "complete" and st["restored_steps"] == 2000 and st["new_steps_this_session"] == 2000
    assert st["n_updates"] == expected_updates and len(st["resumes"]) == 1
    steps = [int(r.split(",")[1]) for r in csv_rows(os.path.join(rd, "eval_points.csv"))]
    assert steps == [0, 2000, 4000]                                  # no duplicate evaluation rows
    again = run_script(script, *args, "--resume", rd, root=tmp_path)
    assert again.returncode != 0 and "already complete" in again.stderr


# --------------------------------------------------------------------------- evaluation isolation
def _rng_snapshot(env):
    return (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate(),
            env.unwrapped.np_random.bit_generator.state)


def test_evaluation_does_not_touch_training_state():
    from stable_baselines3 import DQN
    from agents.common.evaluation import DRQNPolicy, SB3Policy, evaluate
    from agents.common.runner import build_env
    from agents.common.seeding import RunRNG
    from agents.drqn.drqn_core import DRQNAgent, DRQNConfig
    cfg = plan.resolve_config("delivery", "dqn", 1, mode="smoke")
    train_env, eval_env = build_env(cfg), build_env(cfg)
    train_env.reset(seed=5)
    model = DQN("MlpPolicy", train_env, seed=0, learning_starts=10, buffer_size=200)
    model.learn(64)
    before = _rng_snapshot(train_env)
    replay_before = model.replay_buffer.size()
    params = [p.detach().clone() for p in model.policy.parameters()]
    rows = evaluate(eval_env, SB3Policy(model), [1_000_000, 1_000_001])
    after = _rng_snapshot(train_env)
    assert len(rows) == 2 and all(r["episode_seed"] >= 1_000_000 for r in rows)
    assert torch.equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    assert before[2] == after[2] and before[3] == after[3]
    assert model.replay_buffer.size() == replay_before
    assert all(torch.equal(a, b) for a, b in zip(params, model.policy.parameters()))
    # same seeds -> same evaluation episodes (paired across algorithms)
    assert evaluate(eval_env, SB3Policy(model), [1_000_000, 1_000_001]) == rows

    rng = RunRNG(1)
    agent = DRQNAgent(train_env.observation_space, 4, DRQNConfig(enc_dim=16, lstm_dim=16, replay_frames=500), "cpu", rng)
    states = [g.bit_generator.state for g in (rng.exploration, rng.replay, rng.numpy)]
    pol = DRQNPolicy(agent)
    evaluate(eval_env, pol, [1_000_000])
    assert [g.bit_generator.state for g in (rng.exploration, rng.replay, rng.numpy)] == states
    pol.reset(None, None)
    assert float(pol.hidden[0].abs().sum()) == 0.0                     # recurrent state reset per episode


# --------------------------------------------------------------------------- launchers
def _bash():
    for cand in (r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")):
        if cand and os.path.exists(cand) and "system32" not in cand.lower():
            return cand
    return None


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no POSIX bash available")


def _fake_python(tmp_path, exit_code):
    """Wrapper used as --python: real Python for the launcher's own queries, fake training runs."""
    real = sys.executable.replace("\\", "/")
    path = tmp_path / "fakepy.sh"
    path.write_text(f"""#!/usr/bin/env bash
if [[ "$1" == "-c" || "$1" == "-m" ]]; then exec "{real}" "$@"; fi
echo "fake run $*"
if [[ "{exit_code}" == "0" ]]; then echo "complete: fake"; exit 0; fi
echo "fake failure" >&2; exit {exit_code}
""", newline="\n")
    return str(path).replace("\\", "/")


def _launch(role, *args):
    script = os.path.join(REPO, "run_scripts", "baseline", f"run_{role}_baselines.sh" if role == "memoryless"
                          else f"run_{role}_memory_baselines.sh")
    return subprocess.run([BASH, script, *args], capture_output=True, text=True, env=ENV)


@needs_bash
def test_launcher_requires_an_explicit_interpreter_and_lists_the_matrix(tmp_path):
    res = _launch("memoryless")
    assert res.returncode == 2 and "--python PATH is required" in res.stderr
    res = _launch("memoryless", "--python", sys.executable.replace("\\", "/"), "--dry-run")
    assert res.returncode == 0, res.stderr
    cmds = [l for l in res.stdout.splitlines() if "train_" in l]
    assert len(cmds) == 25
    assert all("--require-gpu" in c for c in cmds if "--env colour" in c)
    assert all("--device cpu" in c for c in cmds if "--env colour" not in c)
    assert sum("--task-state" in c for c in cmds) == 10
    assert all("--wandb-project new_experiments" in c and "--wandb-mode online" in c for c in cmds)


@needs_bash
def test_launcher_reports_failures_and_exits_nonzero(tmp_path):
    fake = _fake_python(tmp_path, 7)
    res = _launch("explicit", "--python", fake, "--mode", "smoke", "--seeds", "1", "--skip-preflight",
                  "--results-root", str(tmp_path / "res").replace("\\", "/"))
    assert res.returncode == 1
    assert "BATCH FAILED: 3 failed run(s)" in res.stderr and "All 3 runs exited 0" not in res.stdout
    (tsv,) = [os.path.join(d, f) for d, _, fs in os.walk(tmp_path / "res") for f in fs if f == "batch_status.tsv"]
    rows = open(tsv, encoding="utf-8").read().strip().splitlines()[1:]
    assert [r.split("\t")[4] for r in rows] == ["7", "7", "7"]
    logs = os.path.dirname(rows[0].split("\t")[7])
    assert len(os.listdir(logs)) == 6                                  # stdout + stderr per run

    ok = _launch("explicit", "--python", _fake_python(tmp_path, 0), "--mode", "smoke", "--seeds", "1",
                 "--skip-preflight", "--results-root", str(tmp_path / "res2").replace("\\", "/"))
    assert ok.returncode == 0 and "All 3 runs exited 0" in ok.stdout


@needs_bash
def test_old_remediation_script_is_deprecated():
    res = subprocess.run([BASH, os.path.join(REPO, "run_scripts", "baseline",
                                             "rerun_drqn_image_and_ppo_rm_continuous.sh")],
                         capture_output=True, text=True)
    assert res.returncode == 2 and "deprecated" in res.stderr
