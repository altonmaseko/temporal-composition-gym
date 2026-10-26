"""Hazardous Delivery semantics (protocol v2 sections 3, 5)."""
import numpy as np
import pytest

from agents.common.env_factory import make_env
from agents.common.seeding import eval_episode_seeds
from temporal_comp_gym.envs.hazardous_delivery.objects import (
    DamageSquare, DeathSquare, Destination, HealthNode, LevelGoal, Package)
from temporal_comp_gym.oracles import DeliveryOracle, run_oracle_episode

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def blank(**overrides):
    """Main-condition env with an empty interior and the agent at (1, 1)."""
    env = make_env("delivery", debug_info=True, overrides=overrides or None)
    env.reset(seed=0)
    core = env.unwrapped
    for i in range(1, core.width - 1):
        for j in range(1, core.height - 1):
            core.grid.set(i, j, None)
    core.agent_pos = (1, 1)
    return env, core


def names(info):
    return [e[0] for e in info["events"]]


def test_main_condition_hides_delivered_count():
    env, core = blank()
    assert env.observation_space.shape == (8,)
    core.grid.set(2, 1, Package()); core.grid.set(3, 1, Destination())
    env.step(RIGHT)
    obs, r, *_ = env.step(RIGHT)
    assert r == 10.0 and core.delivered_packages == 1
    # x, y, 4 sensors, current square, health -- no delivered count
    assert obs.tolist()[:2] == [3.0, 1.0] and obs[7] == core.max_health


def test_lava_is_not_premature_goal():
    env, core = blank()
    core.grid.set(1, 2, DeathSquare())
    _, r, term, trunc, info = env.step(DOWN)
    assert names(info) == ["lava"] and r == -50.0 and term and not trunc
    assert info["termination_cause"] == "lava" and info["is_success"] is False

    env, core = blank()
    core.grid.set(2, 1, LevelGoal())
    _, r, term, _, info = env.step(RIGHT)
    assert names(info) == ["goal_contact"] and r == -50.0 and term
    assert info["termination_cause"] == "premature_goal"


def test_scripted_success_rewards_and_history():
    env, core = blank()
    row = [Package(), Destination(), Package(), Destination(), Package(), Destination(), LevelGoal()]
    for x, obj in enumerate(row, start=2):
        core.grid.set(x, 1, obj)
    rewards, evs, states = [], [], []
    for _ in row:
        _, r, term, _, info = env.step(RIGHT)
        rewards.append(r); evs.append(names(info)); states.append((info["debug"]["delivered"], info["debug"]["carried"]))
    assert rewards == [0, 10, 0, 10, 0, 10, 50]
    assert evs[0] == ["package_contact", "pickup"] and evs[1] == ["destination_contact", "delivery"]
    assert states == [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 0)]
    assert term and info["termination_cause"] == "success" and info["is_success"]


def test_capacity_and_empty_handed_contacts():
    env, core = blank()
    core.grid.set(2, 1, Package()); core.grid.set(3, 1, Package()); core.grid.set(1, 2, Destination())
    env.step(RIGHT)
    _, r, _, _, info = env.step(RIGHT)            # second package while carrying one
    assert names(info) == ["package_contact"] and r == 0.0
    assert core.grid.get(3, 1) is not None and core.carried_packages == 1

    env, core = blank()
    core.grid.set(1, 2, Destination())
    _, r, _, _, info = env.step(DOWN)             # destination while empty-handed
    assert names(info) == ["destination_contact"] and r == 0.0 and core.grid.get(1, 2) is not None


def test_damage_fatal_damage_and_healing():
    env, core = blank()
    core.grid.set(2, 1, DamageSquare())
    _, r, term, _, info = env.step(RIGHT)
    assert names(info) == ["damage"] and r == -1.0 and not term and core.health == 9

    env, core = blank()
    core.health = 1
    core.grid.set(2, 1, DamageSquare())
    _, r, term, _, info = env.step(RIGHT)
    assert names(info) == ["damage", "health_depleted"] and r == -51.0 and term
    assert info["termination_cause"] == "damage_death"

    env, core = blank()
    core.health = 3
    core.grid.set(2, 1, HealthNode())
    _, r, _, _, info = env.step(RIGHT)
    assert names(info) == ["heal"] and core.health == core.max_health and core.grid.get(2, 1) is None


def test_truncation_is_not_termination():
    env, core = blank(max_steps=5)
    for t in range(5):
        _, _, term, trunc, info = env.step(UP)       # bump into the top wall
    assert trunc and not term and info["termination_cause"] == "timeout" and not info["is_success"]


def test_seeded_reset_and_rollout_are_deterministic():
    def rollout(seed):
        env = make_env("delivery")
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(7)
        trace = [obs.copy()]
        grid = env.unwrapped.grid.encode().copy()
        for _ in range(60):
            obs, r, term, trunc, info = env.step(int(rng.integers(4)))
            trace.append(np.append(obs, [r, term, trunc]))
            if term or trunc:
                obs, _ = env.reset()
                trace.append(obs.copy())
        return grid, np.concatenate(trace)
    g1, t1 = rollout(123)
    g2, t2 = rollout(123)
    g3, _ = rollout(124)
    assert np.array_equal(g1, g2) and np.array_equal(t1, t2)
    assert not np.array_equal(g1, g3)


def test_every_generated_layout_is_solvable():
    env = make_env("delivery")
    for seed in range(1000):
        env.reset(seed=seed)
        assert env.unwrapped.layout_is_solvable()


def test_unsupported_task_dependencies_raise():
    with pytest.raises(NotImplementedError):
        make_env("delivery", overrides={"task_dependencies": True})


@pytest.mark.slow
def test_oracle_solves_final_eval_seeds():
    env = make_env("delivery", debug_info=True)
    oracle = DeliveryOracle(env)
    res = [run_oracle_episode(env, oracle, s) for s in eval_episode_seeds("final", 200)]
    assert all(r["success"] for r in res)
    assert max(r["length"] for r in res) < 500
