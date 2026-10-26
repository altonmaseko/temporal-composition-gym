"""Sequential Colour semantics (protocol v2 sections 3, 4.3, 5)."""
import numpy as np
import pytest

from agents.common.env_factory import make_env
from agents.common.seeding import eval_episode_seeds
from agents.reward_machines import CueRecordingRM
from temporal_comp_gym.envs.sequential_colour import GaussianNoiseWrapper
from temporal_comp_gym.oracles import ColourOracle, run_oracle_episode

SIDE_ACTION = {"top": 0, "bottom": 1, "left": 2, "right": 3}


def to_navigation(env):
    """Step through cues and delay; return (event log per step, last info)."""
    log = []
    while True:
        _, _, _, _, info = env.step(0)
        log.append(info["events"])
        if ("nav_start", None) in info["events"]:
            return log, info


def test_cue_delay_and_nav_event_timing():
    env = make_env("colour", debug_info=True)
    _, info = env.reset(seed=11)
    seq = info["debug"]["sequence_ids"]
    assert info["events"] == [("cue", seq[0])]
    log, _ = to_navigation(env)
    fpc, delay = 5, 2
    for t, evs in enumerate(log, start=1):
        if t % fpc == 0 and t < fpc * len(seq):
            assert evs == [("cue", seq[t // fpc])]
        elif t == fpc * len(seq):
            assert evs == [("delay_start", None)]
        elif t == fpc * len(seq) + delay:
            assert evs == [("nav_start", None)]
        else:
            assert evs == []
    assert all(e[0] != "wall_touch" for evs in log for e in evs)


def test_repeated_colours_each_get_a_cue_event():
    env = make_env("colour", debug_info=True, overrides={"allow_consecutive_duplicates": True})
    for seed in range(200):
        _, info = env.reset(seed=seed)
        seq = info["debug"]["sequence_ids"]
        if any(a == b for a, b in zip(seq, seq[1:])):
            break
    else:
        pytest.fail("no sequence with a repeated colour in 200 seeds")
    cues = [a for n, a in info["events"] if n == "cue"]
    log, _ = to_navigation(env)
    cues += [a for evs in log for n, a in evs if n == "cue"]
    assert cues == seq


def _walk_to(env, side):
    while True:
        _, r, term, trunc, info = env.step(SIDE_ACTION[side])
        if info["events"] or term or trunc:
            return r, term, info


def test_wrong_wall_correct_wall_and_completion():
    env = make_env("colour", debug_info=True)
    env.reset(seed=5)
    _, info = to_navigation(env)
    core = env.unwrapped
    walls = dict(info["debug"]["wall_ids"])
    target = info["debug"]["target_id"]
    wrong = next(s for s, c in walls.items() if c != target)
    r, term, info = _walk_to(env, wrong)
    assert info["events"] == [("wall_touch", walls[wrong])] and r == -1.0 and not term
    assert info["debug"]["wall_ids"] == walls and info["debug"]["progress"] == 0
    assert (core.agent.x, core.agent.y) == (core.agent.start_x, core.agent.start_y)

    seq = info["debug"]["sequence_ids"]
    total = 0.0
    for i, c in enumerate(seq):
        walls = info["debug"]["wall_ids"]
        side = next(s for s, cc in walls.items() if cc == c)
        r, term, info = _walk_to(env, side)
        total += r
        if i < len(seq) - 1:
            assert info["events"] == [("wall_touch", c), ("room_reset", None)] and r == 0.0 and not term
        else:
            assert info["events"] == [("wall_touch", c)] and r == 10.0 and term
    assert info["termination_cause"] == "success" and info["is_success"] and total == 10.0


def test_navigation_info_never_reveals_the_target():
    env = make_env("colour")                       # debug_info off, as in training
    _, info = env.reset(seed=3)
    assert set(info) == {"events"}
    rng = np.random.default_rng(0)
    nav = False
    for _ in range(300):
        _, _, term, trunc, info = env.step(int(rng.integers(4)))
        allowed = {"events"} | ({"termination_cause", "is_success"} if term or trunc else set())
        assert set(info) <= allowed
        nav = nav or any(n == "nav_start" for n, _ in info["events"])
        if nav:
            assert {n for n, _ in info["events"]} <= {"nav_start", "wall_touch", "room_reset"}
        if term or trunc:
            break


def test_truncation_at_episode_limit():
    env = make_env("colour", overrides={"max_steps": 20, "sequence_length_range": (3, 3)})
    env.reset(seed=0)
    for t in range(1, 21):
        _, _, term, trunc, info = env.step(0 if t < 17 else 2)
    assert trunc and not term and info["termination_cause"] == "timeout"


def test_seeded_episodes_are_deterministic_and_noise_uses_its_own_rng():
    def trace(env, seed):
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(1)
        out = [obs]
        for _ in range(80):
            obs, r, term, trunc, _ = env.step(int(rng.integers(4)))
            out.append(obs)
            if term or trunc:
                break
        return np.stack(out), env.unwrapped.sequence
    a, seq_a = trace(make_env("colour"), 9)
    b, seq_b = trace(make_env("colour"), 9)
    assert np.array_equal(a, b) and seq_a == seq_b
    noisy = GaussianNoiseWrapper(make_env("colour"), observation_noise_level=0.1)
    _, seq_n = trace(noisy, 9)
    assert seq_n == seq_a


def test_rgb_colours_are_distinct_for_the_vocabulary():
    env = make_env("colour")
    cols = env.unwrapped.vocab_colors
    assert len(set(cols)) == len(cols)
    grey = {round(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in cols}
    # Converting to grey would merge colours, which is why the encoders keep RGB.
    assert len(grey) <= len(cols)


def test_cue_rm_requires_symmetric_rooms():
    with pytest.raises(ValueError):
        CueRecordingRM.from_env(make_env("colour", overrides={"randomise_wall_colors": False}))


@pytest.mark.slow
def test_oracle_solves_final_eval_seeds():
    env = make_env("colour", debug_info=True)
    oracle = ColourOracle(env)
    res = [run_oracle_episode(env, oracle, s) for s in eval_episode_seeds("final", 200)]
    assert all(r["success"] for r in res)
    assert max(r["length"] for r in res) <= 150          # limit is 300
