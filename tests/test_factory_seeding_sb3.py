"""Environment factory, seed sets, import source, and SB3 transition handling."""
import os

import numpy as np
import pytest

import agents
import temporal_comp_gym
from agents.common import seeding
from agents.common.env_factory import BASELINE_ENV_KWARGS, env_kwargs, make_env
from agents.reward_machines import RewardMachineWrapper, make_reward_machine

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_code_is_imported_from_this_checkout():
    for mod in (temporal_comp_gym, agents):
        assert os.path.commonpath([os.path.abspath(mod.__file__), REPO_ROOT]) == REPO_ROOT


def test_frozen_main_conditions():
    assert BASELINE_ENV_KWARGS["delivery"]["include_inventory_in_state"] is False      # D1
    assert BASELINE_ENV_KWARGS["maintenance"]["distractor_stations"] is True            # D2
    assert [BASELINE_ENV_KWARGS[k]["max_steps"] for k in ("delivery", "colour", "maintenance")] == [500, 300, 4000]
    shapes = {k: make_env(k).observation_space.shape for k in ("delivery", "colour", "maintenance")}
    assert shapes == {"delivery": (8,), "colour": (84, 84, 3), "maintenance": (28,)}
    with pytest.raises(KeyError):
        env_kwargs("delivery", overrides={"not_a_setting": 1})
    assert env_kwargs("delivery", "inventory_visible")["include_inventory_in_state"] is True


def test_seed_sets_are_disjoint():
    train_env = {seeding.train_env_seed(s) for r in seeding.TRAIN_SEED_RANGES.values() for s in r}
    blocks = [set(range(b, b + seeding.EVAL_BLOCK_SIZE)) for b in seeding.EVAL_EPISODE_SEED_BASE.values()]
    for i, a in enumerate(blocks):
        assert not a & train_env
        for b in blocks[i + 1:]:
            assert not a & b
    ranges = list(seeding.TRAIN_SEED_RANGES.values())
    for i, a in enumerate(ranges):
        for b in ranges[i + 1:]:
            assert not set(a) & set(b)
    assert set(seeding.BASELINE_TRAIN_SEEDS) <= set(seeding.TRAIN_SEED_RANGES["baseline"])


def test_run_streams_are_reproducible_and_independent():
    a, b, c = seeding.RunRNG(1), seeding.RunRNG(1), seeding.RunRNG(2)
    assert a.replay.integers(1 << 30) == b.replay.integers(1 << 30)
    assert a.exploration.integers(1 << 30) != c.exploration.integers(1 << 30)
    names = ["torch", "agent_numpy", "python", "action_space", "replay", "exploration"]
    assert len({seeding.stream_seed(1, n) for n in names}) == len(names)


def test_sb3_receives_truncation_flags_from_internal_limits():
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([lambda: make_env("delivery", overrides={"max_steps": 3})])
    venv.seed(0)
    venv.reset()
    for _ in range(3):
        _, _, dones, infos = venv.step(np.array([0]))
    assert dones[0] and infos[0]["TimeLimit.truncated"] and "terminal_observation" in infos[0]


def test_ppo_rm_and_task_state_controls_build_and_train():
    from stable_baselines3 import DQN, PPO
    base = make_env("maintenance")
    env = RewardMachineWrapper(base, make_reward_machine("maintenance", base), obs_mode="append", reward_mode="rm")
    assert env.observation_space.shape == (28 + 8,)
    PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0).learn(128)

    base = make_env("delivery")
    env = RewardMachineWrapper(base, make_reward_machine("delivery", base), obs_mode="append")
    assert env.observation_space.shape == (8 + 7,)
    DQN("MlpPolicy", env, learning_starts=32, buffer_size=500, seed=0).learn(100)

    base = make_env("colour")
    env = RewardMachineWrapper(base, make_reward_machine("colour", base), obs_mode="dict")
    obs, _ = env.reset(seed=0)
    assert obs["image"].shape == (84, 84, 3) and obs["task_state"].sum() == 1.0
    DQN("MultiInputPolicy", env, learning_starts=32, buffer_size=200, seed=0).learn(80)
