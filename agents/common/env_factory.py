"""Single place that builds the protocol-v2 environments.

Every algorithm in an environment receives the same base environment from here; agent
specific wrappers (reward-machine state, task-state features) are added on top by the
trainers. The baseline settings are the frozen ones in run_scripts/baseline/BASELINE_PROTOCOL.md.
"""
import copy
import os

import gymnasium as gym

import temporal_comp_gym  # noqa: F401  (registers the environments)
from temporal_comp_gym.envs.deferred_maintenance.wrappers import MaintenanceScaledObservation

PROTOCOL_VERSION = "v2"

ENV_IDS = {
    "delivery": "TemporalComp/HazardousDelivery-v0",
    "colour": "TemporalComp/SequentialColour-v0",
    "maintenance": "TemporalComp/DeferredMaintenance-v0",
}

# Main ("non-Markovian") condition per environment, protocol v2 section 3.1.
BASELINE_ENV_KWARGS = {
    "delivery": {
        "grid_size": 10,
        "percent_death_squares": 0.05,
        "percent_damage_squares": 0.05,
        "randomise_start": True,
        "randomise_goal": True,
        "num_packages": 3,
        "carrying_capacity": 1,
        "health_regeneration_nodes": 2,
        "task_dependencies": False,
        "include_health_in_state": True,
        "include_inventory_in_state": False,   # D1: delivered count hidden
        "max_steps": 500,
        "resample_unsolvable": True,
    },
    "colour": {
        "sequence_length_range": (3, 5),
        "retention_delay_steps": 2,
        "allow_consecutive_duplicates": False,
        "colour_vocabulary_size": 6,
        "randomise_wall_colors": True,
        "hard_death_mode": False,
        "infinite_mode": False,
        "reward_density": "sparse",
        "frames_per_colour": 5,
        "max_steps": 300,
    },
    "maintenance": {
        "max_stages": 4,
        "readiness_delay_range": (1, 2),
        "max_code_types": 3,
        "max_active_requests": 3,
        "codes_per_collection_stage": 1,
        "distractor_stations": True,           # D2: stations for non-issued types too
        "required_hover_duration": 2.0,
        "spatial_sensor_type": "local_grid",
        "gravity_scale": 500.0,
        "max_steps": 4000,
    },
}

# Optional observability ablations (not part of the 55-run baseline matrix).
ABLATIONS = {
    "delivery": {
        "inventory_visible": {"include_inventory_in_state": True},
        "health_hidden": {"include_health_in_state": False},
    },
    "colour": {},
    "maintenance": {
        "no_distractors": {"distractor_stations": False},
    },
}


def env_kwargs(env_key, variant="main", overrides=None):
    """Resolved constructor kwargs for an environment variant (a fresh copy)."""
    if env_key not in ENV_IDS:
        raise KeyError(f"unknown env_key {env_key!r}; expected one of {sorted(ENV_IDS)}")
    kwargs = copy.deepcopy(BASELINE_ENV_KWARGS[env_key])
    if variant != "main":
        if variant not in ABLATIONS[env_key]:
            raise KeyError(f"unknown variant {variant!r} for {env_key}")
        kwargs.update(ABLATIONS[env_key][variant])
    if overrides:
        unknown = set(overrides) - set(kwargs)
        if unknown:
            raise KeyError(f"overrides for unknown settings: {sorted(unknown)}")
        kwargs.update(overrides)
    return kwargs


def make_env_from_kwargs(env_key, kwargs, debug_info=False):
    """Build from a run's resolved env kwargs (JSON lists are fine where tuples are expected)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    kwargs = dict(kwargs)
    assert "max_episode_steps" not in kwargs
    kwargs["debug_info"] = debug_info
    env = gym.make(ENV_IDS[env_key], **kwargs)
    if env_key == "maintenance":
        env = MaintenanceScaledObservation(env)
    return env


def make_env(env_key, variant="main", overrides=None, debug_info=False, render_mode=None):
    """Base environment every algorithm in `env_key` receives.

    Maintenance gets the fixed observation scaling (no running statistics). No extra
    TimeLimit is added: each environment truncates internally at its own limit.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    kwargs = env_kwargs(env_key, variant, overrides)
    # Never pass `max_episode_steps`: gym.make would consume it and add its own TimeLimit,
    # bypassing the env's internal limit and its timeout cause. The envs use `max_steps`.
    assert "max_episode_steps" not in kwargs
    kwargs["debug_info"] = debug_info
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(ENV_IDS[env_key], **kwargs)
    if env_key == "maintenance":
        env = MaintenanceScaledObservation(env)
    return env
