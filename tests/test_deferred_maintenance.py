"""Deferred Maintenance semantics (protocol v2 sections 3, 5)."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agents.common.env_factory import make_env
from agents.common.seeding import eval_episode_seeds
from temporal_comp_gym.envs.deferred_maintenance.core import CELL_PATH, DeferredMaintenanceEnv
from temporal_comp_gym.envs.deferred_maintenance.objects import CodeZone, MaintenanceStation
from temporal_comp_gym.oracles import MaintenanceOracle, run_oracle_episode

HOVER = np.array([0.25, 0.0], dtype=np.float32)    # thrust exactly cancels gravity


def fresh(seed=4, **overrides):
    env = make_env("maintenance", debug_info=True, overrides=overrides or None)
    env.reset(seed=seed)
    return env, env.unwrapped


def teleport(core, pos):
    core.rocket.body.position = (float(pos[0]), float(pos[1]))
    core.rocket.body.velocity = (0.0, 0.0)
    core.rocket.body.angular_velocity = 0.0
    core.rocket.body.angle = 0.0


def centre(core, cell):
    return ((cell[0] + 0.5) * core.cell_size, (cell[1] + 0.5) * core.cell_size)


def station(core, k):
    return next(e for e in core.entities if isinstance(e, MaintenanceStation) and e.color_code == k)


def hover(env, n):
    out = []
    for _ in range(n):
        out.append(env.step(HOVER))
    return out


def test_gymnasium_env_checker_passes():
    check_env(DeferredMaintenanceEnv(distractor_stations=True), skip_render_check=True)


def test_seeded_layout_and_rollout_are_deterministic():
    def run(seed):
        env, core = fresh(seed)
        layout = [row[:] for row in core.stage_layout]
        sched = (core.readiness_target, list(core.issued_codes))
        rng = np.random.default_rng(3)
        trace = []
        for _ in range(300):
            obs, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
            trace.append(np.append(obs, r))
            if term or trunc:
                break
        return layout, sched, np.stack(trace)
    a, b, c = run(42), run(42), run(43)
    assert a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2])
    assert a[0] != c[0]


def test_score_is_the_running_sum_of_rewards():
    env, core = fresh(1)
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(600):
        _, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        total += r
        assert core.score == pytest.approx(total)
        if term or trunc:
            break


def test_schedule_is_solvable_by_construction():
    env = make_env("maintenance")
    core = env.unwrapped
    for seed in range(300):
        env.reset(seed=seed)
        r, issued = core.readiness_target, core.issued_codes
        assert 1 <= r <= 2 and len(issued) == r and len(set(issued)) == r
        assert len(issued) <= core.max_active_requests
        assert core.station_types == [1, 2, 3]            # distractors on (D2)
        for s in range(core.max_stages):
            layout, _ = core._generate_procedural_stage(s)
            cells = [v for row in layout for v in row]
            for k in core.station_types:
                assert cells.count(f"station_{k}") == 1
            codes = sorted(int(v.split("_")[1]) for v in cells if isinstance(v, str) and v.startswith("code_"))
            assert codes == sorted(core.stage_code_types.get(s, []))


def test_invalid_or_unimplemented_configurations_raise():
    with pytest.raises(ValueError):
        DeferredMaintenanceEnv(readiness_delay_range=(1, 4), max_stages=4)
    with pytest.raises(ValueError):
        DeferredMaintenanceEnv(readiness_delay_range=(0, 2))
    with pytest.raises(ValueError):
        DeferredMaintenanceEnv(readiness_delay_range=(1, 3), max_stages=5, max_code_types=2)
    with pytest.raises(NotImplementedError):
        DeferredMaintenanceEnv(duration_matching_mode=True)


def test_macro_waypoint_points_to_next_path_tile():
    env = make_env("maintenance")
    obs, _ = env.reset(seed=8)
    core = env.unwrapped
    expected = (np.array(centre(core, core.path_cells[1])) - np.array(core.rocket.body.position)) / core.cell_size
    assert np.allclose(obs[15:17], expected, atol=1e-5) and np.abs(expected).sum() > 0


def test_code_collection_event_and_tile_conversion():
    env, core = fresh(4)
    code = next(e for e in core.entities if isinstance(e, CodeZone))
    k = code.color_code
    teleport(core, code.body.position)
    obs, r, _, _, info = env.step(HOVER)
    assert ("code_collected", k) in info["events"] and info["debug"]["held_codes"] == [k]
    cell = (code.rect[0] // core.cell_size, code.rect[1] // core.cell_size)
    assert core.cell_types[cell] == CELL_PATH and core.uncollected_codes == 0
    assert obs[17] == pytest.approx(k / 3)               # floor code while touching it


def test_premature_success_and_wrong_service():
    env, core = fresh(4)
    k = core.stage_code_types[0][0]
    teleport(core, station(core, k).body.position)
    steps = hover(env, 120)
    assert all(s[1] == 0.0 and not s[4]["events"] for s in steps[:119])
    assert steps[119][4]["events"] == [("service_attempt", k)] and steps[119][1] == -10.0   # not ready

    env, core = fresh(4)
    core.readiness, core.held_codes = 1.0, [k]
    teleport(core, station(core, k).body.position)
    *_, last = hover(env, 120)
    assert last[1] == 100.0 and core.held_codes == [] and core.serviced_codes == [k]

    env, core = fresh(4)
    core.readiness, core.held_codes = 1.0, []
    teleport(core, station(core, k).body.position)
    *_, last = hover(env, 120)
    assert last[4]["events"] == [("service_attempt", k)] and last[1] == -10.0          # wrong station


def test_exit_outcomes():
    env, core = fresh(4)
    teleport(core, centre(core, core.path_cells[-1]))
    _, r, term, _, info = env.step(HOVER)
    assert ("stage_exit_incomplete", 0) in info["events"] and r == -1000.0 and term
    assert info["termination_cause"] == "uncollected_codes_at_exit"

    for held, reward, cause in (([], 1000.0, "success"), ([2], -1000.0, "held_codes_at_final_exit")):
        env, core = fresh(4)
        code = next(e for e in core.entities if isinstance(e, CodeZone))
        teleport(core, code.body.position)
        env.step(HOVER)
        core.stages_traversed = core.max_stages - 1
        core.held_codes = list(held)
        teleport(core, centre(core, core.path_cells[-1]))
        _, r, term, _, info = env.step(HOVER)
        assert [e[0] for e in info["events"]][-2:] == ["stage_exit", "final_exit"]
        assert r == reward and term and info["termination_cause"] == cause


def test_truncation_at_episode_limit():
    env, core = fresh(4, max_steps=30)
    *_, last = hover(env, 30)
    assert last[3] and not last[2] and last[4]["termination_cause"] == "timeout"


def test_readiness_switches_on_when_entering_the_readiness_stage():
    env, core = fresh(0)
    oracle = MaintenanceOracle(env)
    oracle.reset()
    target = core.readiness_target
    while True:
        _, r, term, trunc, info = env.step(oracle.act())
        names = [e[0] for e in info["events"]]
        assert "service_attempt" not in names or r >= 100.0 - 6.0      # oracle never mis-services
        if "readiness_on" in names:
            assert ("stage_exit", target - 1) in info["events"] and core.stages_traversed == target
        if term or trunc:
            break
    assert info["is_success"]


@pytest.mark.slow
def test_oracle_success_and_horizon():
    env = make_env("maintenance", debug_info=True)
    oracle = MaintenanceOracle(env)
    res = [run_oracle_episode(env, oracle, s) for s in eval_episode_seeds("final", 40)]
    assert np.mean([r["success"] for r in res]) >= 0.95
    assert np.percentile([r["length"] for r in res], 95) <= env.unwrapped.max_steps / 2
