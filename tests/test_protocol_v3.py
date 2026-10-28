"""Protocol v3 (decisions D5-D9): v2 unchanged, v3 differences, noisy evaluation, extra logging."""
import json
import os
import random

import numpy as np
import pytest
import torch

from agents.common import baseline_plan as plan
from agents.common.evaluation import EpsilonPolicy, SB3Policy, evaluate
from agents.common.run_io import read_json
from agents.common.runner import build_env
from tests.test_pipeline import TINY, csv_rows, run_script


def v3_run_dirs(root):
    import glob
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(str(root), "v3", "smoke", "*", "*", "seed*__*", "identity.json")))


def _diff(a, b, path=""):
    out = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            out.update(_diff(a.get(k), b.get(k), f"{path}.{k}" if path else k))
    elif a != b:
        out[path] = (a, b)
    return out


def test_v2_plan_and_hashes_are_unchanged():
    assert plan.main(["write", "--check"]) == 0
    stored = json.load(open(plan.PLAN_PATH, encoding="utf-8"))
    for r in stored["runs"]:
        assert plan.config_hash(plan.resolve_config(r["env"], r["algo"], r["seed"])) == r["config_hash"]


@pytest.mark.parametrize("env_key,algo", [("delivery", "dqn"), ("colour", "qrm_cue"), ("maintenance", "ppo"),
                                          ("maintenance", "recurrent_ppo"), ("maintenance", "ppo_rm")])
def test_v3_differs_from_v2_only_by_the_accepted_decisions(env_key, algo):
    v2 = plan.resolve_config(env_key, algo, 1)
    v3 = plan.resolve_config(env_key, algo, 1, protocol_version="v3")
    d = _diff(v2, v3)
    expected = {"protocol_version", "wandb.group", "eval.secondary"}
    if env_key == "maintenance":
        expected.add("algo.hyperparams.gamma")
        assert v3["algo"]["hyperparams"]["gamma"] == 0.999
    assert set(d) == expected
    assert plan.config_hash(v2) != plan.config_hash(v3)
    with pytest.raises(ValueError):
        plan.resolve_config(env_key, algo, 1, protocol_version="v9")


def test_logged_dqn_trains_identically_to_sb3_dqn():
    from stable_baselines3 import DQN
    from agents.common.sb3_extras import LoggedDQN
    cfg = plan.resolve_config("delivery", "dqn", 1, mode="smoke")
    params = []
    for cls in (DQN, LoggedDQN):
        env = build_env(cfg)
        m = cls("MlpPolicy", env, seed=3, learning_starts=50, buffer_size=500, train_freq=4, device="cpu")
        m.learn(400)
        params.append([p.detach().clone() for p in m.policy.parameters()])
        if cls is LoggedDQN:
            vals = dict(m.logger.name_to_value)
    assert all(torch.equal(a, b) for a, b in zip(*params))
    assert np.isfinite(vals["train/q_mean"]) and vals["train/grad_norm"] >= 0


def _snapshot(env):
    return (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate(),
            env.unwrapped.np_random.bit_generator.state)


@pytest.mark.parametrize("env_key,algo", [("delivery", "dqn"), ("maintenance", "ppo")])
def test_noisy_evaluation_is_reproducible_and_isolated(env_key, algo):
    from stable_baselines3 import DQN, PPO
    cfg = plan.resolve_config(env_key, algo, 1, mode="smoke", protocol_version="v3")
    train_env, eval_env = build_env(cfg), build_env(cfg)
    train_env.reset(seed=5)
    if algo == "dqn":
        model = DQN("MlpPolicy", train_env, seed=0, learning_starts=10, buffer_size=200, device="cpu")
        noisy = EpsilonPolicy(SB3Policy(model), 4, 0.05)
    else:
        model = PPO("MlpPolicy", train_env, seed=0, n_steps=64, batch_size=32, device="cpu")
        noisy = SB3Policy(model, deterministic=False)
    seeds = [1_000_000, 1_000_001]
    before = _snapshot(train_env)
    a = evaluate(eval_env, noisy, seeds, rng_offset=cfg["eval"]["secondary"]["rng_offset"])
    after = _snapshot(train_env)
    b = evaluate(eval_env, noisy, seeds, rng_offset=cfg["eval"]["secondary"]["rng_offset"])
    assert a == b
    assert torch.equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    assert before[2] == after[2] and before[3] == after[3]


def test_episode_counters_for_stage_exits_and_wall_touches():
    from temporal_comp_gym.oracles import ORACLES
    for key, field, expect in (("maintenance", "stage_exits", 4), ("colour", "wall_touches", None)):
        cfg = plan.resolve_config(key, "ppo" if key == "maintenance" else "dqn", 1, protocol_version="v3")
        env = build_env(cfg)
        oracle = ORACLES[key](env)
        env.reset(seed=2_000_000)
        oracle.reset()
        seq_len = len(env.unwrapped.sequence) if key == "colour" else None
        while True:
            _, _, term, trunc, info = env.step(oracle.act())
            if term or trunc:
                rec = info["episode_record"]
                break
        assert rec["success"]
        assert rec[field] == (expect if expect is not None else seq_len)
        other = "wall_touches" if field == "stage_exits" else "stage_exits"
        assert rec[other] is None


@pytest.mark.slow
@pytest.mark.parametrize("script,env_key,extra", [("agents/train_dqn.py", "delivery", TINY),
                                                  ("agents/train_ppo.py", "maintenance", [])])
def test_v3_smoke_run_writes_secondary_evaluation(tmp_path, script, env_key, extra):
    res = run_script(script, "--env", env_key, "--seed", "1", "--mode", "smoke", "--protocol", "v3",
                     "--wandb-mode", "disabled", *extra, root=tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    (rd,) = v3_run_dirs(tmp_path)
    assert os.sep.join(["v3", "smoke", env_key]) in rd
    cfg = read_json(os.path.join(rd, "config.resolved.json"))
    assert cfg["protocol_version"] == "v3" and "secondary" in cfg["eval"]
    if env_key == "maintenance":
        assert cfg["algo"]["hyperparams"]["gamma"] == 0.999
    n_pts = cfg["eval"]["n_points"] + 1
    assert len(csv_rows(os.path.join(rd, "eval_points_noisy.csv"))) == n_pts
    assert len(csv_rows(os.path.join(rd, "eval_final_noisy.csv"))) == cfg["eval"]["final_episodes"]
    summary = read_json(os.path.join(rd, "final_summary.json"))
    assert summary["final_eval_noisy"]["n_episodes"] == cfg["eval"]["final_episodes"]
    if env_key == "delivery":
        header = open(os.path.join(rd, "train_updates.csv"), encoding="utf-8").readline()
        rows = csv_rows(os.path.join(rd, "train_updates.csv"))
        q_col = header.strip().split(",").index("q_mean")
        assert any(r.split(",")[q_col] for r in rows)                    # LoggedDQN q_mean recorded
    else:
        header = open(os.path.join(rd, "train_episodes.csv"), encoding="utf-8").readline()
        assert "stage_exits" in header
