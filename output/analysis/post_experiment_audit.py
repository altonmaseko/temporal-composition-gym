"""Read-only diagnostic probes for the current implementation; no training or run data.

Run from the repository root with the temporal_gym Python interpreter.
All custom layouts and state changes exist only in this diagnostic process.
"""

import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agents"))

import gymnasium as gym
import numpy as np

import temporal_comp_gym
from temporal_comp_gym.envs.deferred_maintenance import core as maintenance
from temporal_comp_gym.envs.hazardous_delivery.objects import DeathSquare
from rm_wrapper import (
    DeferredMaintenanceRMWrapper,
    HazardousDeliveryRMWrapper,
    SequentialColourRMWrapper,
)
from qrm.train_qrm import DeliveryRewardMachine, SequentialColourRewardMachine
from ppo_rm.train_ppo_rm import PPORMObservationWrapper


def fingerprint(env):
    core = env.unwrapped
    state = [
        (type(e).__name__, list(e.rect), int(getattr(e, "color_code", 0)))
        for e in core.entities
    ]
    state.append(("readiness_target", int(core.readiness_target)))
    return hashlib.sha256(json.dumps(state).encode()).hexdigest()[:16]


def seed_probe():
    np.random.seed(123)
    env = gym.make("TemporalComp/DeferredMaintenance-v0")
    env.reset(seed=123)
    first = fingerprint(env)
    env.reset(seed=123)
    second = fingerprint(env)
    env.close()
    np.random.seed(123)
    env = gym.make("TemporalComp/DeferredMaintenance-v0")
    env.reset(seed=123)
    replay = fingerprint(env)
    env.close()
    assert first != second and first == replay
    return {
        "first": first,
        "same_reset_seed_again": second,
        "reseed_global_and_reconstruct": replay,
        "reset_seed_alone_repeats": first == second,
        "controlled_global_replay_repeats": first == replay,
    }


def toggle_probe():
    results = {}
    for flag in ("distractor_stations", "duration_matching_mode"):
        states = []
        for value in (False, True):
            np.random.seed(123)
            env = gym.make("TemporalComp/DeferredMaintenance-v0", **{flag: value})
            obs, _ = env.reset(seed=123)
            state = [fingerprint(env), obs.tolist(), env.unwrapped.required_hover_duration]
            for _ in range(12):
                obs, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
                state.append([obs.tolist(), float(reward), terminated, truncated])
            states.append(state)
            env.close()
        assert states[0] == states[1]
        results[flag] = {"identical_layout_observations_and_rewards": True}
    return results


def lava_probe():
    env = HazardousDeliveryRMWrapper(gym.make("TemporalComp/HazardousDelivery-v0"))
    env.reset(seed=123)
    core = env.unwrapped
    core.agent_pos = (1, 1)
    core.grid.set(1, 2, DeathSquare())
    rm = DeliveryRewardMachine(env)
    _, reward, terminated, _, info = env.step(1)
    comparisons = []
    for u in range(rm.num_states):
        emitted = rm.step(u, info["propositions"])
        corrected_label = rm.step(u, {"stepped_on_lava", "died"})
        assert emitted == corrected_label
        comparisons.append({"u": u, "emitted": emitted, "corrected": corrected_label})
    env.close()
    return {
        "environment_reward": reward,
        "terminated": terminated,
        "emitted_propositions": sorted(info["propositions"]),
        "identical_rm_outcomes_for_all_states": comparisons,
    }


def sequential_probe():
    env = SequentialColourRMWrapper(
        gym.make("TemporalComp/SequentialColour-v0", sequence_length_range=(3, 3))
    )
    env.reset(seed=123)
    rm = SequentialColourRewardMachine(env)
    u = rm.get_initial_state()
    events = []
    env_total = 0.0
    rm_total = 0.0
    direction = {"top": 0, "bottom": 1, "left": 2, "right": 3}
    for _ in range(300):
        core = env.unwrapped
        before_phase = core.current_phase
        before_idx = core.current_seq_idx
        if before_phase == "navigation":
            target = core.sequence[before_idx]
            wall = next(w for w, color in core.wall_colors.items() if color == target)
            action = direction[wall]
        else:
            action = 0
        _, reward, terminated, truncated, info = env.step(action)
        u, rm_reward, rm_done = rm.step(u, info["propositions"])
        env_total += reward
        rm_total += rm_reward
        if info["propositions"] or reward:
            events.append({
                "step": core.step_count,
                "phase_before": before_phase,
                "index_before": before_idx,
                "index_after": core.current_seq_idx,
                "propositions": sorted(info["propositions"]),
                "environment_reward": reward,
                "rm_reward": rm_reward,
                "rm_state_after": u,
                "rm_done": rm_done,
                "environment_terminated": terminated,
            })
        if terminated or truncated:
            break
    env.close()
    assert events[0]["phase_before"] == "memorisation"
    assert events[0]["propositions"] == ["correct_wall"]
    assert env_total == 10.0 and rm_total == 5.0
    return {"events": events, "environment_return": env_total, "rm_return": rm_total}


def maintenance_rm_class():
    # Load the actual nested class without executing the training main function.
    source = Path("agents/ppo_rm/train_ppo_rm.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "MaintenanceRewardMachine"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<existing maintenance RM>", "exec"), namespace)
    return namespace["MaintenanceRewardMachine"]


def readiness_horizon_probe():
    original = maintenance.CUSTOM_STAGES
    results = []
    try:
        maintenance.CUSTOM_STAGES = [[[4, 1, 1, 5]]] * 4
        for target in range(1, 9):
            env = gym.make(
                "TemporalComp/DeferredMaintenance-v0", gravity_scale=0.0,
                readiness_delay_range=(target, target + 1), max_stages=4,
            )
            env.reset(seed=123)
            core = env.unwrapped
            core.held_codes = [1]  # Outstanding obligation, retained across stages.
            signals = [core.readiness]
            for _ in range(4):
                exit_zone = next(e for e in core.entities if isinstance(e, maintenance.StageExit))
                core.rocket.body.position = exit_zone.body.position
                core.rocket.body.velocity = (0, 0)
                _, reward, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
                signals.append(core.readiness)
                if terminated:
                    break
            results.append({
                "target_stages": target,
                "readiness_initial_then_after_exits": signals,
                "final_reward": reward,
                "terminated": terminated,
            })
            assert any(signals) == (target < 4)
            env.close()
    finally:
        maintenance.CUSTOM_STAGES = original
    return results


def maintenance_inventory_probe(rm_class):
    original = maintenance.CUSTOM_STAGES
    cases = []
    try:
        maintenance.CUSTOM_STAGES = [[[4, 1, "station_1", 1, 5]]]
        for held in ([1], [2]):
            env = PPORMObservationWrapper(
                DeferredMaintenanceRMWrapper(gym.make(
                    "TemporalComp/DeferredMaintenance-v0", gravity_scale=0.0,
                    required_hover_duration=1 / 60, penalty_off_path=0.0,
                )), rm_class,
            )
            env.reset(seed=123)
            core = env.unwrapped
            core.readiness = 1.0
            env.u = 1
            core.held_codes = list(held)
            station = next(e for e in core.entities if isinstance(e, maintenance.MaintenanceStation))
            core.rocket.body.position = station.body.position
            core.rocket.body.velocity = (0, 0)
            before = env._get_obs(env.get_wrapper_attr("observation")(core._get_obs()))
            _, rm_reward, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
            cases.append({"held_codes": held, "augmented_observation": before.tolist(), "reward": rm_reward})
            env.close()
        assert cases[0]["augmented_observation"] == cases[1]["augmented_observation"]
        assert cases[0]["reward"] == 100.0 and cases[1]["reward"] == -10.0
    finally:
        maintenance.CUSTOM_STAGES = original
    return {"identical_observations_with_different_required_history": True, "cases": cases}


def score_reward_probe(rm_class):
    original = maintenance.CUSTOM_STAGES
    try:
        maintenance.CUSTOM_STAGES = [[[4, "station_1", 5]]]
        env = DeferredMaintenanceRMWrapper(gym.make(
            "TemporalComp/DeferredMaintenance-v0", gravity_scale=0.0,
            required_hover_duration=1 / 60, max_stages=1, penalty_off_path=0.0,
        ))
        env.reset(seed=123)
        core = env.unwrapped
        core.readiness = 1.0
        core.held_codes = [1]
        station = next(e for e in core.entities if isinstance(e, maintenance.MaintenanceStation))
        exit_zone = next(e for e in core.entities if isinstance(e, maintenance.StageExit))
        core.rocket.body.position = ((station.body.position.x + exit_zone.body.position.x) / 2, station.body.position.y)
        core.rocket.body.velocity = (0, 0)
        _, raw_reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        _, rm_reward, _ = rm_class(env).step(1, info["propositions"])
        result = {"simultaneous_service_and_final_exit": True, "environment_reward": raw_reward,
                  "score_delta_rm_reward": rm_reward, "terminated": terminated,
                  "propositions": sorted(info["propositions"])}
        assert raw_reward == 1100.0 and rm_reward == 1200.0
        env.close()
        return result
    finally:
        maintenance.CUSTOM_STAGES = original


def run_probes():
    rm_class = maintenance_rm_class()
    return {
        "scope": "Current local implementation; synthetic diagnostics, no experiment data or training",
        "audit_date": "2026-10-24",
        "python_executable": sys.executable,
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "agents/rm_wrapper.py", "agents/qrm/train_qrm.py",
                "agents/ppo_rm/train_ppo_rm.py",
                "temporal_comp_gym/envs/deferred_maintenance/core.py",
                "temporal_comp_gym/envs/sequential_colour/core.py",
            )
        },
        "maintenance_seeding": seed_probe(),
        "maintenance_toggles": toggle_probe(),
        "hazardous_lava_label": lava_probe(),
        "sequential_rm": sequential_probe(),
        "maintenance_readiness_horizon": readiness_horizon_probe(),
        "maintenance_inventory_memory": maintenance_inventory_probe(rm_class),
        "maintenance_rm_score_reward_divergence": score_reward_probe(rm_class),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run_probes(), indent=2))
