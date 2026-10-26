"""The protocol-v2 baseline plan: run matrix, budgets, seeds, evaluation and hyperparameters.

This module is the single source of truth for what a baseline run is. A run's resolved
configuration (resolve_config) is a plain JSON-able dict; its hash identifies the run's
scientific settings. The launchers obtain their run lists from here, and
run_scripts/baseline/baseline_plan.json is written from here.

    python -m agents.common.baseline_plan write            # (re)write baseline_plan.json
    python -m agents.common.baseline_plan list --role memoryless [--mode full]
"""
import argparse
import copy
import hashlib
import json
import os
import sys

from agents.common.env_factory import ENV_IDS, PROTOCOL_VERSION, env_kwargs
from agents.common.seeding import BASELINE_TRAIN_SEEDS, TRAIN_SEED_RANGES, train_env_seed
from agents.drqn.drqn_core import DRQN_DEFAULTS
from agents.qrm.qrm_core import QRM_DEFAULTS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLAN_PATH = os.path.join(REPO_ROOT, "run_scripts", "baseline", "baseline_plan.json")
PHASE = "baseline"
DEFAULT_WANDB_PROJECT = "new_experiments"
FORBIDDEN_WANDB_PROJECTS = {"temporal-composition-gym"}      # the pilot project

# --------------------------------------------------------------------------- algorithms
ALGORITHMS = {
    #  name           family        machine role   envs                          entry script                         implementation
    "dqn":           {"family": "memoryless", "role": "memoryless", "envs": ["delivery", "colour"], "script": "agents/train_dqn.py", "impl": "sb3"},
    "ppo":           {"family": "memoryless", "role": "memoryless", "envs": ["maintenance"], "script": "agents/train_ppo.py", "impl": "sb3"},
    "drqn":          {"family": "implicit", "role": "implicit", "envs": ["delivery", "colour"], "script": "agents/drqn/train_drqn.py", "impl": "custom"},
    "recurrent_ppo": {"family": "implicit", "role": "implicit", "envs": ["maintenance"], "script": "agents/train_recurrent_ppo.py", "impl": "sb3"},
    "qrm":           {"family": "explicit", "role": "explicit", "envs": ["delivery"], "script": "agents/qrm/train_qrm.py", "impl": "custom"},
    "qrm_cue":       {"family": "explicit", "role": "explicit", "envs": ["colour"], "script": "agents/qrm/train_qrm.py", "impl": "custom"},
    "ppo_rm":        {"family": "explicit", "role": "explicit", "envs": ["maintenance"], "script": "agents/ppo_rm/train_ppo_rm.py", "impl": "sb3"},
    "dqn_ts":        {"family": "control", "role": "memoryless", "envs": ["delivery", "colour"], "script": "agents/train_dqn.py", "impl": "sb3"},
}

# Main matrix per machine role, in run order within a seed (seed-major order).
ROLE_MAIN_RUNS = {
    "memoryless": [("delivery", "dqn"), ("colour", "dqn"), ("maintenance", "ppo")],
    "implicit": [("delivery", "drqn"), ("colour", "drqn"), ("maintenance", "recurrent_ppo")],
    "explicit": [("delivery", "qrm"), ("colour", "qrm_cue"), ("maintenance", "ppo_rm")],
}
# Task-state controls (D4) run on the memoryless machine after all of its main runs.
ROLE_CONTROL_RUNS = {"memoryless": [("delivery", "dqn_ts"), ("colour", "dqn_ts")], "implicit": [], "explicit": []}
ROLES = tuple(ROLE_MAIN_RUNS)

# --------------------------------------------------------------------------- budgets / eval
BUDGETS = {"delivery": 1_000_000, "colour": 3_000_000, "maintenance": 2_048_000}

EVAL = {
    # periodic evaluation at step 0 and every budget/n_points; final evaluation after the budget
    "delivery": {"n_points": 20, "periodic_episodes": 50, "final_episodes": 200},
    "colour": {"n_points": 20, "periodic_episodes": 50, "final_episodes": 200},
    "maintenance": {"n_points": 20, "periodic_episodes": 20, "final_episodes": 100},
}
CHECKPOINT_POINTS = 10          # restart checkpoint and policy snapshot every budget/10
TRAIN_LOG_POINTS = 200          # aggregated training metrics to W&B every budget/200

# --------------------------------------------------------------------------- hyperparameters
# Untuned common defaults, protocol v2 section 6.2. SB3 keyword arguments are passed through.
_DQN_COMMON = {"gamma": 0.99, "buffer_size": 100_000, "train_freq": 4, "gradient_steps": 1, "batch_size": 32,
               "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "exploration_fraction": 0.2,
               "max_grad_norm": 10.0}
_PPO_COMMON = {"n_steps": 2048, "batch_size": 64, "n_epochs": 10, "learning_rate": 3e-4, "gamma": 0.99,
               "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5}
SB3_HYPERPARAMS = {
    ("dqn", "delivery"): {"policy": "MlpPolicy", **_DQN_COMMON, "learning_rate": 5e-4, "learning_starts": 10_000,
                          "target_update_interval": 2_000, "policy_kwargs": {"net_arch": [128, 128]}},
    ("dqn", "colour"): {"policy": "CnnPolicy", **_DQN_COMMON, "learning_rate": 1e-4, "learning_starts": 20_000,
                        "target_update_interval": 10_000,
                        "policy_kwargs": {"net_arch": [256], "features_extractor_kwargs": {"features_dim": 512}}},
    ("dqn_ts", "delivery"): {"policy": "MlpPolicy", **_DQN_COMMON, "learning_rate": 5e-4, "learning_starts": 10_000,
                             "target_update_interval": 2_000, "policy_kwargs": {"net_arch": [128, 128]}},
    ("dqn_ts", "colour"): {"policy": "MultiInputPolicy", **_DQN_COMMON, "learning_rate": 1e-4, "learning_starts": 20_000,
                           "target_update_interval": 10_000,
                           "policy_kwargs": {"net_arch": [256], "features_extractor_kwargs": {"cnn_output_dim": 512}}},
    ("ppo", "maintenance"): {"policy": "MlpPolicy", **_PPO_COMMON,
                             "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}},
    ("ppo_rm", "maintenance"): {"policy": "MlpPolicy", **_PPO_COMMON,
                                "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}},
    ("recurrent_ppo", "maintenance"): {"policy": "MlpLstmPolicy", **_PPO_COMMON,
                                       "policy_kwargs": {"net_arch": {"pi": [128], "vf": [128]}, "lstm_hidden_size": 128,
                                                         "n_lstm_layers": 1, "shared_lstm": False,
                                                         "enable_critic_lstm": True}},
}


def custom_hyperparams(algo, env_key):
    if algo == "drqn":
        return DRQN_DEFAULTS[env_key].to_dict()
    if algo in ("qrm", "qrm_cue"):
        return QRM_DEFAULTS[env_key].to_dict()
    raise KeyError(algo)


def hyperparams(algo, env_key):
    if ALGORITHMS[algo]["impl"] == "sb3":
        return copy.deepcopy(SB3_HYPERPARAMS[(algo, env_key)])
    return custom_hyperparams(algo, env_key)


# --------------------------------------------------------------------------- smoke mode
# Reduced settings for pipeline checks only. Outputs go to results/<v>/smoke and W&B group
# "<phase>-<v>-smoke"; they are never part of the production manifest.
SMOKE = {
    "budget": {"delivery": 4_000, "colour": 3_000, "maintenance": 4_096},
    "eval": {"n_points": 2, "periodic_episodes": 2, "final_episodes": 3},
    "checkpoint_points": 2,
    "train_log_points": 10,
    "env_overrides": {"maintenance": {"max_steps": 600}},
    "hyperparams": {
        "sb3_dqn": {"learning_starts": 500, "target_update_interval": 250, "buffer_size": 5_000, "exploration_fraction": 0.5},
        "sb3_ppo": {"n_steps": 512, "batch_size": 64, "n_epochs": 2},
        "custom": {"learning_starts": 500, "target_update": 250, "replay_frames": 5_000, "eps_fraction": 0.5},
        "drqn_colour": {"batch_size": 2, "train_len": 16, "burn_in": 8},
        "qrm_batch": {"batch_size": 8},
    },
}


def _smoke_hyperparams(algo, env_key, hp):
    s = SMOKE["hyperparams"]
    if ALGORITHMS[algo]["impl"] == "sb3":
        hp.update(s["sb3_ppo"] if "ppo" in algo else s["sb3_dqn"])
    else:
        hp.update(s["custom"])
        if algo == "drqn" and env_key == "colour":
            hp.update(s["drqn_colour"])
        if algo in ("qrm", "qrm_cue"):
            hp.update(s["qrm_batch"])
    return hp


# --------------------------------------------------------------------------- resolution
def algo_for(script_algo, env_key):
    """train_qrm.py serves QRM (delivery) and QRM-Cue (colour)."""
    if script_algo == "qrm" and env_key == "colour":
        return "qrm_cue"
    return script_algo


def _set_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"override {dotted!r}: unknown key {k!r}")
        node = node[k]
    if not isinstance(node, dict) or keys[-1] not in node:
        raise KeyError(f"override {dotted!r}: unknown key {keys[-1]!r} (overrides may only change existing settings)")
    node[keys[-1]] = value


def parse_override(text):
    """'a.b=VALUE' with VALUE parsed as JSON when possible (so 1e-4, true, [1,2] work)."""
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like key.path=value")
    key, raw = text.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def resolve_config(env_key, algo, seed, mode="full", overrides=None, wandb_project=DEFAULT_WANDB_PROJECT,
                   wandb_entity=None):
    """The complete resolved configuration of one run (JSON-able dict)."""
    if algo not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {algo!r}")
    if env_key not in ALGORITHMS[algo]["envs"]:
        raise ValueError(f"{algo} is not part of the plan for {env_key}")
    if mode not in ("full", "smoke"):
        raise ValueError("mode must be 'full' or 'smoke'")
    if wandb_project in FORBIDDEN_WANDB_PROJECTS:
        raise ValueError(f"W&B project {wandb_project!r} is the pilot project; baseline runs must not log there")
    seed = int(seed)
    if seed not in TRAIN_SEED_RANGES["baseline"]:
        raise ValueError(f"seed {seed} is outside the baseline training seed range {TRAIN_SEED_RANGES['baseline']}")

    budget = BUDGETS[env_key]
    ev = dict(EVAL[env_key])
    ckpt_points, log_points = CHECKPOINT_POINTS, TRAIN_LOG_POINTS
    ekw = env_kwargs(env_key)
    hp = hyperparams(algo, env_key)
    if mode == "smoke":
        budget = SMOKE["budget"][env_key]
        ev.update(SMOKE["eval"])
        ckpt_points, log_points = SMOKE["checkpoint_points"], SMOKE["train_log_points"]
        ekw.update(SMOKE["env_overrides"].get(env_key, {}))
        hp = _smoke_hyperparams(algo, env_key, hp)

    cfg = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": PHASE,
        "mode": mode,
        "env": {"key": env_key, "id": ENV_IDS[env_key], "variant": "main", "kwargs": _jsonable(ekw)},
        "algo": {"name": algo, "family": ALGORITHMS[algo]["family"], "impl": ALGORITHMS[algo]["impl"],
                 "hyperparams": _jsonable(hp)},
        "seed": {"train_seed": seed, "train_env_reset_seed": train_env_seed(seed)},
        "budget": {"env_steps": int(budget)},
        "eval": {
            "n_points": int(ev["n_points"]),
            "every_steps": int(budget // ev["n_points"]),
            "periodic_episodes": int(ev["periodic_episodes"]),
            "final_episodes": int(ev["final_episodes"]),
            "periodic_seed_block": "periodic",
            "final_seed_block": "final",
            "policy": "greedy (epsilon=0 / deterministic=True); recurrent state reset at episode start",
        },
        "checkpoint": {"every_steps": int(budget // ckpt_points), "save_replay": True},
        "logging": {"train_log_every_steps": max(1, int(budget // log_points))},
        "overrides": {},
        "wandb": {"project": wandb_project, "entity": wandb_entity,
                  "group": f"{PHASE}-{PROTOCOL_VERSION}" + ("-smoke" if mode == "smoke" else "")},
    }
    for key, value in (overrides or {}).items():
        if key.split(".")[0] in ("protocol_version", "phase", "mode", "seed", "wandb", "overrides"):
            raise KeyError(f"override {key!r} is not allowed; use the dedicated option")
        _set_path(cfg, key, value)
        cfg["overrides"][key] = value
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    b = cfg["budget"]["env_steps"]
    hp = cfg["algo"]["hyperparams"]
    if cfg["algo"]["impl"] == "sb3" and "n_steps" in hp:
        if b % hp["n_steps"]:
            raise ValueError(f"budget {b} is not a multiple of n_steps {hp['n_steps']} (SB3 would round it up)")
    if cfg["algo"]["impl"] == "sb3" and "train_freq" in hp and b % hp["train_freq"]:
        raise ValueError(f"budget {b} is not a multiple of train_freq {hp['train_freq']}")
    if cfg["eval"]["every_steps"] * cfg["eval"]["n_points"] != b:
        raise ValueError("eval.every_steps x eval.n_points must equal the budget")
    if b % cfg["checkpoint"]["every_steps"]:
        raise ValueError("checkpoint.every_steps must divide the budget")
    if cfg["eval"]["final_episodes"] < 1 or cfg["eval"]["periodic_episodes"] < 1:
        raise ValueError("evaluation needs at least one episode")


def hashed_part(cfg):
    """Settings that define the experiment (W&B destination is excluded)."""
    return {k: v for k, v in cfg.items() if k != "wandb"}


def config_hash(cfg):
    blob = json.dumps(hashed_part(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- run lists
def role_runs(role, mode="full", seeds=BASELINE_TRAIN_SEEDS, include_controls=True):
    """Ordered (env, algo, seed) for a machine role: main runs seed-major, then controls."""
    runs = [(e, a, s) for s in seeds for e, a in ROLE_MAIN_RUNS[role]]
    if include_controls:
        runs += [(e, a, s) for s in seeds for e, a in ROLE_CONTROL_RUNS[role]]
    return runs


def expected_runs(mode="full", seeds=BASELINE_TRAIN_SEEDS):
    out = []
    for role in ROLES:
        for env_key, algo, seed in role_runs(role, mode, seeds):
            cfg = resolve_config(env_key, algo, seed, mode)
            out.append({"role": role, "env": env_key, "algo": algo, "seed": seed, "family": ALGORITHMS[algo]["family"],
                        "script": ALGORITHMS[algo]["script"], "task_state": algo == "dqn_ts",
                        "config_hash": config_hash(cfg)})
    return out


def build_plan():
    runs = expected_runs("full")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "phase": PHASE,
        "description": "Resolved protocol-v2 baseline plan (written by agents/common/baseline_plan.py).",
        "seeds": list(BASELINE_TRAIN_SEEDS),
        "budgets_env_steps": BUDGETS,
        "evaluation": EVAL,
        "checkpoint_points": CHECKPOINT_POINTS,
        "wandb": {"default_project": DEFAULT_WANDB_PROJECT, "group": f"{PHASE}-{PROTOCOL_VERSION}",
                  "auth": "wandb login on each machine (~/.netrc); the code never reads the API key"},
        "roles": {role: [f"{e}/{a}/seed{s}" for e, a, s in role_runs(role)] for role in ROLES},
        "run_count": {"total": len(runs), "main": sum(r["family"] != "control" for r in runs),
                      "control": sum(r["family"] == "control" for r in runs)},
        "runs": runs,
        "resolved_configs": {f"{r['env']}/{r['algo']}/seed{r['seed']}": resolve_config(r["env"], r["algo"], r["seed"])
                             for r in runs},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write run_scripts/baseline/baseline_plan.json")
    w.add_argument("--out", default=PLAN_PATH)
    w.add_argument("--check", action="store_true", help="fail if the file on disk differs instead of writing")
    ls = sub.add_parser("list", help="print 'env algo seed script' lines for a machine role")
    ls.add_argument("--role", choices=ROLES, required=True)
    ls.add_argument("--mode", choices=("full", "smoke"), default="full")
    ls.add_argument("--seeds", type=int, nargs="+", default=list(BASELINE_TRAIN_SEEDS))
    ls.add_argument("--no-controls", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "write":
        text = json.dumps(build_plan(), indent=2, sort_keys=False) + "\n"
        if args.check:
            with open(args.out, encoding="utf-8") as f:
                if f.read() != text:
                    print(f"{args.out} is out of date; run: python -m agents.common.baseline_plan write", file=sys.stderr)
                    return 1
            print(f"{args.out} is up to date")
            return 0
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"wrote {args.out}")
        return 0
    for env_key, algo, seed in role_runs(args.role, args.mode, args.seeds, not args.no_controls):
        print(env_key, algo, seed, ALGORITHMS[algo]["script"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
