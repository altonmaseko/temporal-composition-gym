"""Stress plan v3 (decision D11): determinism, verbatim tuned learners, seeds, paths, regressions."""
import copy
import json
from pathlib import Path

import pytest

from agents.common import baseline_plan as base
from agents.common.env_factory import (ABLATIONS, BASELINE_ENV_KWARGS, STRESS_AXES, STRESS_LEVELS, env_kwargs,
                                       stress_env_kwargs)
from agents.common.seeding import eval_episode_seeds
from agents.common.stress_io import run_parent, safe_root
from agents.common.stress_plan import (CELLS, DEFAULT_SEEDS, PLAN_PATH, RESULTS, ROLES, STRESS_SEEDS, build_plan,
                                       digest, load_plan, load_selected, resolve_config, role_queues, run_config)
from agents.common.tuning_io import check_path
from agents.common.tuning_plan import ROOT

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def plan():
    return load_plan()


def _selected_file():
    return json.loads((ROOT / "run_scripts/tuning/results/v3/full/selected_configs.json").read_text(encoding="utf-8"))


def _trials_file():
    return json.loads((ROOT / "run_scripts/tuning/trials_v3.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- plan
def test_plan_file_matches_generator_and_is_deterministic(plan):
    from run_scripts.stress.generate_plan import main
    assert build_plan() == build_plan() == plan
    assert main(["--check"]) == 0
    body = {k: v for k, v in plan.items() if k != "plan_hash"}
    assert plan["plan_hash"] == digest(body)
    assert plan["run_count"] == len(plan["runs"]) == 154
    keys = {(r["env"], r["algo"], r["level"], r["seed"]) for r in plan["runs"]}
    assert len(keys) == 154
    expected = {(e, a, "default", s) for e, a in CELLS for s in DEFAULT_SEEDS}
    expected |= {(e, a, lv, s) for e, a in CELLS for lv in ("easy", "hard1", "hard2") for s in STRESS_SEEDS}
    assert keys == expected
    for r in plan["runs"]:
        assert r["config_hash"] == base.config_hash(r["config"])
        assert r["config"]["stress"]["level"] == r["level"]
        assert r["device"] == ("cuda" if r["env"] == "colour" else "cpu")
    assert plan["threads"] == 1


def test_check_reports_differing_fields(tmp_path, capsys):
    from run_scripts.stress.generate_plan import main
    bad = copy.deepcopy(load_plan())
    bad["runs"][3]["config"]["algo"]["hyperparams"]["gamma"] = 0.98
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    assert main(["--check", "--out", str(path)]) == 1
    err = capsys.readouterr().err
    assert "/runs[3]/config/algo/hyperparams/gamma: 0.98 in file vs 0.99 expected" in err


def test_hyperparameters_are_verbatim_from_the_tuning_selection(plan):
    sel, trials = _selected_file(), _trials_file()
    cells = {(c["env"], c["algo"]): c for c in trials["cells"]}
    assert plan["tuning_plan_hash"] == sel["plan_hash"] == trials["plan_hash"]
    for r in plan["runs"]:
        chosen = sel["selected"][f"{r['env']}/{r['algo']}"]
        hp = r["config"]["algo"]["hyperparams"]
        # exact JSON values and types, not approximate floats
        assert json.dumps(hp, sort_keys=True) == json.dumps(chosen["hyperparams"], sort_keys=True)
        assert r["candidate"] == chosen["candidate"] == r["config"]["stress"]["tuned_candidate"]
        cand = cells[(r["env"], r["algo"])]["candidates"][chosen["candidate"]]["hyperparams"]
        assert json.dumps(hp, sort_keys=True) == json.dumps(cand, sort_keys=True)
        assert r["config"]["budget"]["env_steps"] == base.BUDGETS[r["env"]]
        assert r["config"]["eval"]["every_steps"] * 20 == base.BUDGETS[r["env"]]
        assert r["config"]["eval"]["secondary"]["value_based_epsilon"] == 0.05
        assert (r["config"]["eval"]["periodic_episodes"], r["config"]["eval"]["final_episodes"]) == \
            ((20, 100) if r["env"] == "maintenance" else (50, 200))


# Fields a stress default-level config may differ from the tuning main-condition config in
# (STRESS_PROTOCOL.md section 2): phase, mode, budget, seed, eval cadence/seed blocks,
# checkpoint/logging intervals, W&B group and the phase-specific blocks.
_PHASE_FIELDS = ("phase", "mode", "budget", "seed", "checkpoint", "logging", "wandb", "tuning", "stress")
_EVAL_PHASE_FIELDS = ("every_steps", "periodic_seed_block", "final_seed_block")


def _strip(cfg):
    c = {k: v for k, v in cfg.items() if k not in _PHASE_FIELDS}
    c["eval"] = {k: v for k, v in cfg["eval"].items() if k not in _EVAL_PHASE_FIELDS}
    return c


def test_default_level_is_exactly_the_main_condition(plan):
    trials = _trials_file()
    tuning_cfg = {(c["env"], c["algo"]): c["candidates"] for c in trials["cells"]}
    for env in BASELINE_ENV_KWARGS:
        assert json.dumps(stress_env_kwargs(env, "default"), sort_keys=True) == \
            json.dumps(env_kwargs(env), sort_keys=True)
    for r in plan["runs"]:
        if r["level"] != "default":
            continue
        cfg = r["config"]
        tcfg = tuning_cfg[(r["env"], r["algo"])][r["candidate"]]["config"]
        assert cfg["env"] == tcfg["env"]                     # id, variant "main", kwargs value for value
        assert base.config_hash(_strip(cfg)) == base.config_hash(_strip(tcfg))
        assert set(cfg) - set(_PHASE_FIELDS) == set(tcfg) - set(_PHASE_FIELDS)


def test_other_levels_change_only_their_axis():
    for env, axis in STRESS_AXES.items():
        main = base._jsonable(env_kwargs(env))
        values = []
        for level in STRESS_LEVELS:
            kw = base._jsonable(stress_env_kwargs(env, level))
            changed = {k for k in kw if kw[k] != main[k]}
            allowed = {axis} | ({"max_steps"} if env == "maintenance" and level.startswith("hard") else set())
            assert changed <= allowed and (level == "default" or axis in changed)
            values.append(kw[axis])
        assert values == {"delivery": [1, 3, 5, 7], "colour": [0, 2, 24, 48],
                          "maintenance": [[1, 1], [1, 2], [2, 2], [3, 3]]}[env]
    assert [stress_env_kwargs("maintenance", lv)["max_steps"] for lv in STRESS_LEVELS] == [4000, 4000, 5000, 5000]
    with pytest.raises(KeyError):
        stress_env_kwargs("delivery", "hard3")


# --------------------------------------------------------------------------- seeds
@pytest.mark.parametrize("level,seed", [("default", 201), ("default", 1), ("default", 101), ("default", 401),
                                        ("easy", 301), ("hard1", 305), ("hard2", 101), ("easy", 401),
                                        ("hard1", 1), ("default", 400), ("easy", 300)])
def test_seeds_rejected_outside_their_phase_and_level(level, seed):
    hp = load_selected()["cells"]["delivery/dqn"]["hyperparams"]
    with pytest.raises(ValueError):
        resolve_config("delivery", "dqn", level, seed, hp, 0)


def test_seed_blocks_and_no_leakage_into_baseline_or_tuning(plan):
    for r in plan["runs"]:
        ev = r["config"]["eval"]
        periodic = eval_episode_seeds(ev["periodic_seed_block"], ev["periodic_episodes"])
        final = eval_episode_seeds(ev["final_seed_block"], ev["final_episodes"])
        lo = 5_000_000 if r["level"] == "default" else 4_000_000
        assert periodic == list(range(lo, lo + ev["periodic_episodes"]))
        assert final == list(range(lo + 10_000, lo + 10_000 + ev["final_episodes"]))
        seed = r["seed"]
        assert (301 <= seed <= 305) if r["level"] == "default" else (201 <= seed <= 203)
        assert r["config"]["seed"]["train_env_reset_seed"] == 10_000_000 + 1000 * seed
    with pytest.raises(ValueError):
        eval_episode_seeds("stress_eval", 10_001)          # periodic seeds stay below the final sub-block
    for seed in (201, 301):
        with pytest.raises(ValueError):
            base.resolve_config("delivery", "dqn", seed, protocol_version="v3")
        from agents.common.tuning_plan import anchor, resolve_config as tuning_resolve
        with pytest.raises(ValueError):
            tuning_resolve("delivery", "dqn", 0, anchor("delivery", "dqn"), seed)


# --------------------------------------------------------------------------- machine plan
def test_machine_queues_cover_every_cell_per_set_and_keep_roles():
    queues = role_queues()
    assert {r: len(q) for r, q in queues.items()} == {"a": 54, "b": 54, "c": 46}
    sets = {}
    for role, q in queues.items():
        for rnd, level, seed, env, algo in q:
            sets.setdefault((level, seed), set()).add((env, algo))
    assert len(sets) == 14 and all(cells == set(CELLS) for cells in sets.values())
    # round 1 = three default seeds; round 2 = one seed of each other level
    def round_sets(n):
        return {(role, lv, s) for role, q in queues.items() for rnd, lv, s, e, a in q if rnd == n}
    assert round_sets(1) == {("a", "default", 301), ("b", "default", 302), ("c", "default", 303)}
    assert {lv for _, lv, _ in round_sets(2)} == {"easy", "hard1", "hard2"}
    assert queues["c"][-2:] == [(5, "hard1", 203, "colour", "drqn"), (5, "hard2", 203, "colour", "drqn")]
    for role in ("a", "b"):
        assert queues[role][-1][3:] == ("colour", "qrm_cue")
    plan = load_plan()
    for r in plan["runs"]:
        assert queues[r["role"]][r["slot"]][1:] == (r["level"], r["seed"], r["env"], r["algo"])


# --------------------------------------------------------------------------- paths
def test_portable_path_lengths(plan):
    for r in plan["runs"]:
        for mode in ("full", "smoke"):
            _, cfg = run_config(plan, r["env"], r["algo"], r["level"], r["seed"], mode)
            path = run_parent(RESULTS, cfg) / "a99"
            check_path(path, 60)
            for leaf in ("checkpoints/restart.tmp/replay.pkl", "metadata_resume_20261030T120000.json",
                         "eval_final_noisy.pre-resume-20261030T120000Z.csv", "status_resume_20261030T120000000000Z.json"):
                assert len(str(path / leaf)) < 240
    check_path(RESULTS / "pf12345678" / "_wb" / ("a" * 12), 90)
    for bad in ("run_scripts/baseline/results", "run_scripts/tuning/results", "analysis"):
        with pytest.raises(ValueError):
            safe_root(ROOT / bad)


# --------------------------------------------------------------------------- regressions
def test_tuning_configs_and_env_variants_unchanged():
    from agents.common.tuning_plan import candidate_config
    from agents.common.tuning_plan import load_plan as load_trials
    saved = json.loads((FIXTURES / "tuning_hashes_pre_stress.json").read_text(encoding="utf-8"))
    trials = load_trials()                                   # == generate_trials.py --check
    assert trials["plan_hash"] == saved["tuning_plan_hash"]
    assert len(saved["hashes"]) == 704
    for key, expected in saved["hashes"].items():
        env, algo, cand, seed, mode = key.split("/")
        assert base.config_hash(candidate_config(trials, env, algo, int(cand), int(seed), mode)) == expected
    for env, variants in saved["env_kwargs"].items():
        assert base._jsonable(env_kwargs(env)) == variants["main"]
        for v in ABLATIONS[env]:
            assert base._jsonable(env_kwargs(env, v)) == variants[v]


def test_baseline_plan_file_unchanged():
    assert base.main(["write", "--check"]) == 0


def test_plan_path_is_checked_in_location():
    assert PLAN_PATH == ROOT / "run_scripts" / "stress" / "stress_plan_v3.json"
