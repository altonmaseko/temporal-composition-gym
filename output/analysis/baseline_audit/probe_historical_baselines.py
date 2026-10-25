"""Synthetic, non-training checks of a supplied historical source snapshot."""

import ast
from contextlib import redirect_stdout
import fnmatch
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
SNAPSHOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SNAPSHOT))
sys.path.insert(1, str(SNAPSHOT / "agents"))
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import gymnasium as gym
import numpy as np
import torch
import temporal_comp_gym
from temporal_comp_gym.envs.deferred_maintenance import core as maintenance
from stable_baselines3.common.monitor import Monitor

assert Path(temporal_comp_gym.__file__).resolve().is_relative_to(SNAPSHOT)


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, SNAPSHOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_node(relative, node_type, name):
    tree = ast.parse((SNAPSHOT / relative).read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, node_type) and n.name == name)


def compile_node(node, namespace):
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<historical source>", "exec"), namespace)
    return namespace[node.name]


def checkpoint_selection():
    results = {}
    for algorithm in ("dqn", "ppo", "recurrent_ppo", "drqn", "qrm"):
        custom = algorithm in ("drqn", "qrm")
        relative = f"agents/{algorithm}/train_{algorithm}.py" if custom else f"agents/train_{algorithm}.py"
        name = "load_latest_checkpoint" if custom else "load_latest_checkpoint_sb3"
        env = "TemporalComp/HazardousDelivery-v0" if algorithm in ("dqn", "drqn", "qrm") else "TemporalComp/DeferredMaintenance-v0"
        step = 900_000 if custom else (1_000_000 if algorithm == "dqn" else 2_000_000)
        suffix = ".pt" if custom else "_steps.zip"
        file = os.path.join("models", f"{algorithm}_{env.split('/')[-1]}_step_{step}{suffix}")
        selected_patterns = []

        def glob_function(pattern):
            selected_patterns.append(pattern)
            return [file] if fnmatch.fnmatch(file, pattern) else []

        payload = {"global_step": step, "model_state_dict": {"inherited": True}, "optimizer_state_dict": {}}
        fake_model = SimpleNamespace(load_state_dict=lambda _: None)
        fake_optimizer = SimpleNamespace(load_state_dict=lambda _: None)
        namespace = {"os": SimpleNamespace(makedirs=lambda *a, **k: None, path=os.path),
                     "glob": SimpleNamespace(glob=glob_function), "re": re,
                     "torch": SimpleNamespace(load=lambda _: payload)}
        loader = compile_node(selected_node(relative, ast.FunctionDef, name), namespace)
        with redirect_stdout(io.StringIO()):
            value = loader("models", env, algorithm, fake_model, fake_optimizer) if custom else loader("models", env, algorithm)
        returned_step = value if custom else value[1]
        assert returned_step == step
        results[algorithm] = {"selected_pattern": selected_patterns[0], "selected_checkpoint": file,
                              "returned_start_step": returned_step, "seed_not_part_of_loader_signature": True,
                              "remaining_steps_at_original_budget": (1_000_000 if "Hazardous" in env else 2_000_000) - returned_step}
    return results


def image_and_replay(drqn):
    env = gym.make("TemporalComp/SequentialColour-v0")
    obs, _ = env.reset(seed=1)
    model = drqn.DRQN(env)
    try:
        with torch.no_grad():
            values, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), None)
        inference = {"works": True, "output_shape": list(values.shape)}
    except RuntimeError as exc:
        inference = {"works": False, "error": str(exc)}
    env.close()
    gate_results = []
    for n in (23, 28, 29):
        buffer = drqn.EpisodicReplayBuffer(10000, 8)
        updates_enabled = 0
        for _ in range(n):
            buffer.add(np.zeros(1), 0, 0.0, np.zeros(1), True)
            updates_enabled += int(len(buffer.buffer) > 32)
        assert updates_enabled == 0
        gate_results.append({"completed_episodes": n, "update_gate_true_count": updates_enabled})
    return {"visual_inference": inference, "replay_gate": gate_results}


def defaults():
    result = {}
    for name in ("HazardousDelivery", "SequentialColour", "DeferredMaintenance"):
        env = gym.make(f"TemporalComp/{name}-v0")
        env.reset(seed=1)
        core = env.unwrapped
        values = {"observation_shape": list(env.observation_space.shape), "registered_max_episode_steps": env.spec.max_episode_steps}
        for key in ("max_steps", "task_dependencies", "retention_delay_steps", "hard_death_mode", "max_stages", "readiness_delay_range"):
            if hasattr(core, key):
                values[key] = getattr(core, key)
        if name == "HazardousDelivery":
            values["health_visible"] = env.get_wrapper_attr("include_health")
            values["delivered_count_visible"] = env.get_wrapper_attr("include_inventory")
        result[name] = values
        env.close()
    return result


def ppo_rm_checks(module, rm_wrappers):
    source = (SNAPSHOT / "agents/ppo_rm/train_ppo_rm.py").read_text(encoding="utf-8")
    if "class MaintenanceRewardMachine" not in source:
        env = rm_wrappers.HazardousDeliveryRMWrapper(gym.make("TemporalComp/DeferredMaintenance-v0"))
        try:
            module.PPORMObservationWrapper(env, module.DeliveryRewardMachine)
            raise AssertionError("Historical initial maintenance RM unexpectedly constructed")
        except AttributeError as exc:
            result = {"maintenance_setup_works": False, "error": str(exc)}
        finally:
            env.close()
        return result
    rm_class = compile_node(selected_node("agents/ppo_rm/train_ppo_rm.py", ast.ClassDef, "MaintenanceRewardMachine"), {})
    original = maintenance.CUSTOM_STAGES
    try:
        maintenance.CUSTOM_STAGES = [[[4, "station_1", 5]]]
        base = gym.make("TemporalComp/DeferredMaintenance-v0", gravity_scale=0.0,
                        required_hover_duration=1 / 60, max_stages=1, penalty_off_path=0.0)
        stats = gym.wrappers.RecordEpisodeStatistics(base)
        augmented = module.PPORMObservationWrapper(rm_wrappers.DeferredMaintenanceRMWrapper(stats), rm_class)
        monitor = Monitor(augmented)
        monitor.reset(seed=1)
        core = monitor.unwrapped
        core.readiness = 1.0
        augmented.u = 1
        core.held_codes = [1]
        station = next(e for e in core.entities if isinstance(e, maintenance.MaintenanceStation))
        exit_zone = next(e for e in core.entities if isinstance(e, maintenance.StageExit))
        core.rocket.body.position = ((station.body.position.x + exit_zone.body.position.x) / 2, station.body.position.y)
        _, reward, terminated, _, info = monitor.step(np.zeros(2, dtype=np.float32))
        raw_return = float(stats.return_queue[-1])
        monitored_return = float(info["episode"]["r"])
        assert raw_return == 1100.0 and reward == 1200.0 and monitored_return == 1200.0 and terminated
        result = {"maintenance_setup_works": True, "rm_states": 2,
                  "raw_environment_episode_return": raw_return, "rm_step_reward": reward,
                  "outer_monitor_episode_return": monitored_return,
                  "monitor_overwrites_inner_raw_episode_return": True}
        monitor.close()
        return result
    finally:
        maintenance.CUSTOM_STAGES = original


def main():
    drqn = load_module("baseline_historical_drqn", "agents/drqn/train_drqn.py")
    ppo_rm = load_module("baseline_historical_ppo_rm", "agents/ppo_rm/train_ppo_rm.py")
    import rm_wrapper
    result = {"snapshot": SNAPSHOT.name, "imported_environment_package": str(temporal_comp_gym.__file__),
              "python": sys.executable, "defaults": defaults(),
              "checkpoint_selection": checkpoint_selection(),
              "drqn": image_and_replay(drqn), "ppo_rm": ppo_rm_checks(ppo_rm, rm_wrapper)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
