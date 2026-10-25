"""Non-training probes of the historical tuning objectives and wrapper factory."""

import ast
from contextlib import redirect_stdout
import fnmatch
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def node(path, name, node_type=ast.FunctionDef):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(item for item in ast.walk(tree) if isinstance(item, node_type) and item.name == name)


def compile_node(item, namespace):
    exec(compile(ast.Module(body=[item], type_ignores=[]), "<historical tuning source>", "exec"), namespace)
    return namespace[item.name]


def objective_race():
    results = {}
    for algorithm, environment in (("qrm", "HazardousDelivery"), ("drqn", "SequentialColour")):
        source = OUT / "snapshots/2e6b0f3/run_scripts/hyperparameter_tuning" / f"tune_{algorithm}.py"
        own = f"runs/TemporalComp_{environment}-v0__{algorithm}__0__100"
        other = f"runs/TemporalComp_{environment}-v0__{algorithm}__1__101"
        extracted = []
        commands = []
        fake_path = SimpleNamespace(dirname=os.path.dirname, abspath=os.path.abspath, join=os.path.join,
                                    getctime=lambda path: {own: 100, other: 101}[path])

        def get_reward(path):
            extracted.append(path)
            return -20.0 if path == own else -5.0

        namespace = {"__file__": str(source), "os": SimpleNamespace(path=fake_path),
                     "glob": SimpleNamespace(glob=lambda _: [own, other]),
                     "subprocess": SimpleNamespace(run=lambda cmd, **kwargs: commands.append(cmd)),
                     "get_latest_reward": get_reward}
        objective = compile_node(node(source, "objective"), namespace)
        trial = SimpleNamespace(number=0, suggest_float=lambda _name, low, high, **kw: low,
                                suggest_categorical=lambda _name, choices: choices[0])
        with redirect_stdout(io.StringIO()):
            score = objective(trial)
        assert extracted == [other] and score == -5.0
        results[algorithm] = {"launched_seed": commands[0][commands[0].index("--seed") + 1],
                              "own_run": own, "own_score": -20.0,
                              "selected_run": extracted[0], "returned_score": score,
                              "cross_trial_score_assignment_reproduced": True}
    return results


def checkpoint_isolation():
    result = {}
    for algorithm in ("qrm", "drqn"):
        source = OUT / "snapshots/2e6b0f3/agents" / algorithm / f"train_{algorithm}.py"
        environment = "HazardousDelivery" if algorithm == "qrm" else "SequentialColour"
        filenames = [os.path.join("models", f"{algorithm}_{environment}-v0_step_900000.pt"),
                     os.path.join("models", f"{algorithm}_{environment}-v0_seed1_step_200000.pt")]
        namespace = {"os": SimpleNamespace(makedirs=lambda *a, **kw: None, path=os.path),
                     "glob": SimpleNamespace(glob=lambda pattern: [name for name in filenames if fnmatch.fnmatch(name, pattern)]),
                     "re": re,
                     "torch": SimpleNamespace(load=lambda _: {"global_step": 200000, "model_state_dict": {}, "optimizer_state_dict": {}})}
        loader = compile_node(node(source, "load_latest_checkpoint"), namespace)
        model = SimpleNamespace(load_state_dict=lambda _: None)
        with redirect_stdout(io.StringIO()):
            other_seed = loader("models", f"TemporalComp/{environment}-v0", algorithm, model, model, 2)
            same_seed = loader("models", f"TemporalComp/{environment}-v0", algorithm, model, model, 1)
        assert other_seed == 0 and same_seed == 200000
        result[algorithm] = {"different_seed_starts_at_zero": True,
                             "unseeded_baseline_checkpoint_ignored": True,
                             "existing_same_seed_still_resumes": True}
    return result


def maintenance_factory():
    snapshot = OUT / "snapshots/07c3656"
    sys.path.insert(0, str(snapshot))
    sys.path.insert(1, str(snapshot / "agents"))
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import gymnasium as gym
    import numpy as np
    import temporal_comp_gym
    from stable_baselines3.common.monitor import Monitor
    from rm_wrapper import DeferredMaintenanceRMWrapper
    from temporal_comp_gym.envs.deferred_maintenance import core as maintenance
    assert Path(temporal_comp_gym.__file__).resolve().is_relative_to(snapshot)
    source = snapshot / "run_scripts/hyperparameter_tuning/tune_ppo_rm.py"
    namespace = {"gym": gym, "np": np, "Monitor": Monitor,
                 "DeferredMaintenanceRMWrapper": DeferredMaintenanceRMWrapper}
    for name, kind in (("MaintenanceRewardMachine", ast.ClassDef), ("PPORMObservationWrapper", ast.ClassDef), ("make_env", ast.FunctionDef)):
        compile_node(node(source, name, kind), namespace)
    env = namespace["make_env"]()
    env.reset(seed=1)
    env.unwrapped.space.gravity = (0, 0)
    for step in range(1, 1001):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        if terminated or truncated:
            break
    assert step == 1000 and truncated and not terminated
    env.close()
    result = {"snapshot_package": str(temporal_comp_gym.__file__),
              "stationary_episode_steps": step, "terminated": bool(terminated), "truncated": bool(truncated)}

    original = maintenance.CUSTOM_STAGES
    try:
        maintenance.CUSTOM_STAGES = [[[4, "station_1", 5]]]
        env = namespace["make_env"]()
        core = env.unwrapped
        core.space.gravity = (0, 0)
        core.required_hover_duration = 1 / 60
        core.max_stages = 1
        core.penalty_off_path = 0.0
        raw_rewards = []
        original_step = core.step

        def tracked_step(action):
            value = original_step(action)
            raw_rewards.append(value[1])
            return value

        core.step = tracked_step
        env.reset(seed=1)
        core.readiness = 1.0
        env.get_wrapper_attr("rm")  # Confirm expected augmentation wrapper is present.
        augmentation = env.env
        augmentation.u = 1
        core.held_codes = [1]
        station = next(e for e in core.entities if isinstance(e, maintenance.MaintenanceStation))
        exit_zone = next(e for e in core.entities if isinstance(e, maintenance.StageExit))
        core.rocket.body.position = ((station.body.position.x + exit_zone.body.position.x) / 2, station.body.position.y)
        _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert raw_rewards == [1100.0] and reward == 1200.0 and info["episode"]["r"] == 1200.0
        result["compound_event_reward"] = {"raw": raw_rewards[0], "modified": reward,
                                          "monitor": info["episode"]["r"]}
        env.close()
    finally:
        maintenance.CUSTOM_STAGES = original
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = {"date": "2026-10-25", "training_executed": False,
              "objective_race": objective_race(), "checkpoint_isolation": checkpoint_isolation(),
              "maintenance_factory": maintenance_factory()}
    (OUT / "tuning_logic_probes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
