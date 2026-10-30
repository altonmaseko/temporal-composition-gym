"""Stress levels (decision D11): validity, solvability, RM == env reward, state sizes, records.

Diagnostic episode seeds only (6,300,000+). The smoke run uses a registered stress entry in
smoke mode, which is a pipeline check and never part of the analysis.
"""
import copy
import csv
import json

import numpy as np
import pytest

from agents.common.env_factory import STRESS_LEVELS, make_env_from_kwargs, stress_env_kwargs
from agents.common.run_io import STRESS_EPISODE_FIELDS, read_json
from agents.common.runner import build_env
from agents.common.stress_plan import load_plan
from agents.common.tracking import EpisodeStats
from agents.reward_machines import CueRecordingRM, DeliveryRM, RewardMachineWrapper, make_reward_machine
from agents.reward_machines import delivery as delivery_rm
from temporal_comp_gym.oracles import ORACLES

SEED0 = 6_300_000
ENVS = ("delivery", "colour", "maintenance")
EPISODES = {"delivery": 20, "colour": 20, "maintenance": 2}


def level_env(env_key, level, debug_info=True):
    return make_env_from_kwargs(env_key, stress_env_kwargs(env_key, level), debug_info=debug_info)


@pytest.mark.parametrize("env_key", ENVS)
@pytest.mark.parametrize("level", STRESS_LEVELS)
def test_level_kwargs_build_and_oracle_solves(env_key, level):
    env = level_env(env_key, level)
    limit = stress_env_kwargs(env_key, level)["max_steps"]
    oracle = ORACLES[env_key](env)
    for i in range(5 if env_key != "maintenance" else 2):
        env.reset(seed=SEED0 + i)
        oracle.reset()
        steps = 0
        while True:
            _, _, term, trunc, info = env.step(oracle.act())
            steps += 1
            if term or trunc:
                break
        assert info["is_success"] and steps <= limit // 2, (env_key, level, steps)


@pytest.mark.parametrize("env_key", ENVS)
@pytest.mark.parametrize("level", STRESS_LEVELS)
@pytest.mark.parametrize("policy", ["oracle", "random"])
def test_rm_reward_terminal_and_state_equal_env_every_step(env_key, level, policy):
    base = level_env(env_key, level)
    env = RewardMachineWrapper(base, make_reward_machine(env_key, base), check_reward=False)
    oracle = ORACLES[env_key](env)
    rng = np.random.default_rng(7)
    steps = 0
    for ep in range(EPISODES[env_key]):
        env.reset(seed=SEED0 + 100 + ep)
        oracle.reset()
        while True:
            if policy == "oracle":
                a = oracle.act()
            elif env_key == "maintenance":
                a = rng.uniform(-1, 1, 2).astype(np.float32)
            else:
                a = int(rng.integers(4))
            _, r, term, trunc, info = env.step(a)
            steps += 1
            rm = info["rm"]
            assert rm["reward"] == pytest.approx(r, abs=1e-6)
            assert rm["terminal"] == term
            if term or trunc:
                break
            d, s = info["debug"], rm["next_state"]
            if env_key == "delivery":
                assert env.rm.states[s] == (d["delivered"], d["carried"])
            elif env_key == "maintenance":
                assert list(s[0]) == sorted(d["held_codes"]) and s[1] == d["stage"]
            elif d["phase"] == "navigation":
                assert list(s[1]) == d["sequence_ids"] and s[2] == d["progress"]
    assert steps > 0


@pytest.mark.parametrize("n", [1, 7])
def test_delivery_counterfactual_validity_per_level(n):
    """As tests/test_reward_machines.py, at the easy and hard2 package counts."""
    base = level_env("delivery", {1: "easy", 7: "hard2"}[n])
    rm = DeliveryRM.from_env(base)
    assert rm.N == n and rm.n_states == 2 * n + 1
    core = base.unwrapped
    rng = np.random.default_rng(n)
    checked = {c: 0 for c in range(delivery_rm.N_CLASSES)}
    for ep in range(60):
        base.reset(seed=SEED0 + 200 + ep)
        for _ in range(200):
            a = int(rng.integers(4))
            d = core.delivered_packages
            outcomes = {}
            for c_cf in range(rm.cap + 1):
                if d + c_cf > rm.N:
                    continue
                clone = copy.deepcopy(core)
                clone.carried_packages = c_cf
                _, _, _, _, inf = clone.step(a)
                outcomes[c_cf] = delivery_rm.classify(inf["events"])
            _, _, term, trunc, info = base.step(a)
            cls = delivery_rm.classify(info["events"])
            checked[cls] += 1
            for c_cf, cls_cf in outcomes.items():
                u_cf = rm.index[(d, c_cf)]
                world_ok = cls not in (delivery_rm.PICKUP, delivery_rm.PKG_CONTACT) or d + c_cf < rm.N
                if world_ok:
                    assert (cls_cf == cls) == bool(rm.valid[u_cf, cls])
                else:
                    assert not rm.valid[u_cf, cls]
            if term or trunc:
                break
    assert checked[delivery_rm.PICKUP] > 0 and checked[delivery_rm.NONE] > 0


@pytest.mark.parametrize("level", STRESS_LEVELS)
def test_colour_cue_rm_accepts_level_and_register_matches_sequence(level):
    env = level_env("colour", level)
    rm = CueRecordingRM.from_env(env)
    assert rm.n_states == 40
    delay = stress_env_kwargs("colour", level)["retention_delay_steps"]
    for i in range(10):
        _, info = env.reset(seed=SEED0 + 300 + i)
        state = rm.initial_state(info["events"])
        L = len(env.unwrapped.sequence)
        step = 0
        while state[0] != "nav":
            _, _, _, _, info = env.step(0)
            step += 1
            names = [e[0] for e in info["events"]]
            assert ("delay_start" in names) == (delay > 0 and step == 5 * L)
            state, _, _ = rm.step(state, info["events"])
        assert step == 5 * L + delay                       # navigation starts after cues + delay
        assert list(state[1]) == info["debug"]["sequence_ids"]


def test_state_sizes_match_each_level():
    plan = load_plan()
    seen = set()
    for r in plan["runs"]:
        key = (r["env"], r["algo"], r["level"])
        if key in seen or r["algo"] not in ("dqn_ts", "qrm", "qrm_cue", "ppo_rm", "ppo", "recurrent_ppo"):
            continue
        seen.add(key)
        env = build_env(r["config"])
        n = r["config"]["env"]["kwargs"].get("num_packages")
        if key[:2] == ("delivery", "dqn_ts"):
            assert env.observation_space.shape == (8 + 2 * n + 1,)
        elif key[:2] == ("delivery", "qrm"):
            assert env.env.rm.n_states == 2 * n + 1
        elif key[:2] == ("colour", "qrm_cue"):
            assert env.env.rm.n_states == 40
        elif key[:2] == ("colour", "dqn_ts"):
            assert env.observation_space["task_state"].shape == (40,)
        elif key[:2] == ("maintenance", "ppo_rm"):
            assert env.observation_space.shape == (36,)
        elif key[0] == "maintenance":
            assert env.observation_space.shape == (28,)
        env.close()
    assert len(seen) == 4 * 7


def test_within_4000_metrics_and_required_count():
    kw = stress_env_kwargs("maintenance", "hard1")
    stats = EpisodeStats(make_env_from_kwargs("maintenance", kw), "maintenance", stress={"within_steps": 4000})
    stats.reset(seed=SEED0 + 400)
    total, ret4000 = 0.0, None
    while True:
        _, r, term, trunc, info = stats.step(np.zeros(2, dtype=np.float32))
        total += r
        if stats.length == 4000:
            ret4000 = total
        if term or trunc:
            break
    rec = info["episode_record"]
    assert rec["required"] == len(stats.env.unwrapped.issued_codes) == 2
    if rec["length"] > 4000:
        assert rec["return_at_4000"] == pytest.approx(ret4000) and rec["success_by_4000"] is False
    else:
        assert rec["return_at_4000"] == pytest.approx(rec["return_raw"])
    # outside the stress phase no extra keys are recorded
    plain = EpisodeStats(make_env_from_kwargs("delivery", stress_env_kwargs("delivery", "default")), "delivery")
    plain.reset(seed=SEED0)
    while True:
        _, _, term, trunc, info = plain.step(0)
        if term or trunc:
            break
    assert not set(STRESS_EPISODE_FIELDS) & set(info["episode_record"])


def test_smoke_run_records_stress_fields_and_resumes(tmp_path):
    from run_scripts.stress.run_stress import main
    args = ["--env", "delivery", "--algo", "dqn", "--level", "easy", "--seed", "201", "--mode", "smoke",
            "--device", "cpu", "--threads", "1", "--wandb-mode", "disabled", "--results-root", str(tmp_path)]
    assert main(args + ["--debug-fail-at-step", "40"]) == 1
    (rd,) = list(tmp_path.glob("v3/smoke/delivery/dqn/easy/s201/a*"))
    assert read_json(str(rd / "status.json"))["state"] == "failed"
    assert main(args + ["--resume", str(rd)]) == 0
    st = read_json(str(rd / "status.json"))
    assert st["state"] == "complete" and st["verified"] and len(st["resumes"]) == 1
    ident = read_json(str(rd / "identity.json"))
    assert ident["phase"] == "stress" and ident["level"] == "easy" and ident["plan_hash"] == load_plan()["plan_hash"]
    meta = read_json(str(rd / "metadata.json"))
    assert meta["threads"]["requested"] == 1
    with open(rd / "eval_final.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["episode_seed"]) for r in rows] == [4_010_000]
    assert set(STRESS_EPISODE_FIELDS) <= set(rows[0]) and rows[0]["required"] == "1"
    cfg = json.loads((rd / "config.resolved.json").read_text(encoding="utf-8"))
    assert cfg["stress"]["level"] == "easy" and cfg["env"]["kwargs"]["num_packages"] == 1
    with pytest.raises(SystemExit):
        main(["--env", "delivery", "--algo", "dqn", "--level", "default", "--seed", "201", "--mode", "smoke",
              "--threads", "1", "--results-root", str(tmp_path)])
