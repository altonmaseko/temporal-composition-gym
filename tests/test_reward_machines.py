"""Reward machines: reward/terminal agreement, task memory, no leakage, counterfactual validity."""
import copy

import numpy as np
import pytest

from agents.common.env_factory import make_env
from agents.reward_machines import CueRecordingRM, DeliveryRM, RewardMachineWrapper, make_reward_machine
from agents.reward_machines import colour as colour_rm
from agents.reward_machines import delivery as delivery_rm
from temporal_comp_gym.oracles import ORACLES

EPISODES = {"delivery": 40, "colour": 40, "maintenance": 4}


def rollouts(key, policy, episodes, seed0=2_000_000):
    base = make_env(key, debug_info=True)
    env = RewardMachineWrapper(base, make_reward_machine(key, base), check_reward=False)
    oracle = ORACLES[key](env)
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        obs, info = env.reset(seed=seed0 + ep)
        oracle.reset()
        yield env, None, info
        while True:
            if policy == "oracle":
                a = oracle.act()
            elif key == "maintenance":
                a = rng.uniform(-1, 1, 2).astype(np.float32)
            else:
                a = int(rng.integers(4))
            obs, r, term, trunc, info = env.step(a)
            yield env, (r, term, trunc), info
            if term or trunc:
                break


@pytest.mark.parametrize("key", ["delivery", "colour", "maintenance"])
@pytest.mark.parametrize("policy", ["oracle", "random"])
def test_rm_reward_terminal_and_state_match_the_env(key, policy):
    for env, step, info in rollouts(key, policy, EPISODES[key]):
        if step is None:
            continue
        r, term, trunc = step
        rm = info["rm"]
        assert rm["reward"] == pytest.approx(r, abs=1e-6)
        assert rm["terminal"] == term
        if term or trunc:
            continue
        d, s = info["debug"], rm["next_state"]
        if key == "delivery":
            assert env.rm.states[s] == (d["delivered"], d["carried"])
        elif key == "maintenance":
            assert list(s[0]) == sorted(d["held_codes"]) and s[1] == d["stage"]
        elif d["phase"] == "navigation":
            assert list(s[1]) == d["sequence_ids"] and s[2] == d["progress"]


def test_rm_memory_goes_beyond_the_observation():
    # Delivery: the main observation has no carried or delivered count, the RM has both.
    base = make_env("delivery")
    assert base.observation_space.shape == (8,)
    assert DeliveryRM.from_env(base).states == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]
    # Maintenance: held codes and stage index are RM state; readiness is in the observation.
    rm = make_reward_machine("maintenance", make_env("maintenance"))
    f = rm.features(((1, 3), 2, 1))
    assert f.tolist() == [1, 0, 1, 0, 0, 1, 0, 1]


def test_colour_register_is_built_from_events_alone():
    env = make_env("colour", debug_info=True)
    _, info = env.reset(seed=21)
    stream = [info["events"]]
    seq = info["debug"]["sequence_ids"]
    while info["debug"]["phase"] != "navigation":
        _, _, _, _, info = env.step(0)
        stream.append(info["events"])
    env.close()
    del env                                         # the RM below never sees the env
    rm = CueRecordingRM(vocab_size=6, max_len=5)
    state = rm.initial_state(stream[0])
    for evs in stream[1:]:
        state, r, term = rm.step(state, evs)
        assert r == 0.0 and not term
    assert state == ("nav", tuple(seq), 0)
    assert rm.head(state) == rm.nav_head(seq[0], len(seq))


def test_old_wrapper_module_is_retired():
    with pytest.raises(ImportError):
        import agents.rm_wrapper  # noqa: F401


# --------------------------------------------------------------------------- counterfactuals
def test_delivery_counterfactual_validity_matches_env_dynamics():
    """For every step and carried count c', the env (cloned, with c') produces the same event
    class iff the RM marks (d, c') valid for that class."""
    base = make_env("delivery", debug_info=True)
    rm = DeliveryRM.from_env(base)
    core = base.unwrapped
    rng = np.random.default_rng(1)
    checked = {c: 0 for c in range(delivery_rm.N_CLASSES)}
    for ep in range(150):
        base.reset(seed=ep)
        for _ in range(200):
            a = int(rng.integers(4))
            d, c = core.delivered_packages, core.carried_packages
            outcomes = {}
            for c_cf in range(rm.cap + 1):
                if d + c_cf > rm.N:
                    continue
                clone = copy.deepcopy(core)
                clone.carried_packages = c_cf
                _, _, _, _, inf = clone.step(a)
                outcomes[c_cf] = delivery_rm.classify(inf["events"])
            _, _, term, trunc, info = base.step(a)
            cls = delivery_rm.classify(info["events"])
            checked[cls] += 1
            for c_cf, cls_cf in outcomes.items():
                u_cf = rm.index[(d, c_cf)]
                # A package on the grid implies d + c' < N in the counterfactual world; if
                # not, that state cannot share this world and must never be updated.
                world_ok = cls not in (delivery_rm.PICKUP, delivery_rm.PKG_CONTACT) or d + c_cf < rm.N
                if world_ok:
                    assert (cls_cf == cls) == bool(rm.valid[u_cf, cls]), (d, c, c_cf, cls, cls_cf)
                else:
                    assert not rm.valid[u_cf, cls]
            if term or trunc:
                break
    for cls in (delivery_rm.PICKUP, delivery_rm.DELIVERY, delivery_rm.DEST_CONTACT, delivery_rm.NONE):
        assert checked[cls] > 0


def test_colour_counterfactual_validity_matches_env_dynamics():
    """In navigation, give a cloned env each counterfactual (target, remaining) and step the
    same action: the outcome equals the real one exactly for the heads the RM marks valid."""
    base = make_env("colour", debug_info=True)
    core = base.unwrapped
    rm = CueRecordingRM.from_env(base)
    rng = np.random.default_rng(2)
    oracle = ORACLES["colour"](base)          # mixed in so that sequences get completed
    seen = {}
    for ep in range(25):
        _, info = base.reset(seed=ep)
        state = rm.initial_state(info["events"])
        for _ in range(300):
            nav = state is not None and state[0] == "nav"
            a = oracle.act() if (ep % 2 == 0 and rng.random() < 0.6) else int(rng.integers(4))
            pre = copy.deepcopy(core) if nav else None
            obs, r, term, trunc, info = base.step(a)
            nxt, _, _ = rm.step(state, info["events"])
            if nav:
                ex = rm.transition_extras(state, info["events"], nxt)
                head = rm.head(state)
                valid, nh, rew, tm = rm.counterfactual({k_: [v] for k_, v in ex.items()}, [head], [rm.head(nxt)])
                assert valid[0, head], "the visited head must always be updated"
                seen[ex["cls"]] = seen.get(ex["cls"], 0) + 1
                if ex["cls"] != colour_rm.NAV_NONE or seen[ex["cls"]] <= 30:
                    k = pre.current_seq_idx
                    seq_ids = [pre.colour_id(c) for c in pre.sequence]
                    c_next = seq_ids[k + 1] if k + 1 < len(seq_ids) else 0
                    for h in np.flatnonzero(rm.is_nav):
                        c_cf, r_cf = int(rm.head_c[h]), int(rm.head_r[h])
                        cf = seq_ids[:k] + [c_cf] + ([c_next] if r_cf >= 2 else []) + [0] * max(0, r_cf - 2)
                        clone = copy.deepcopy(pre)
                        clone.sequence = [pre.vocab_colors[i] for i in cf]
                        o2, r2, t2, _, _ = clone.step(a)
                        same = np.array_equal(o2, obs) and r2 == r and t2 == term
                        assert same == bool(valid[0, h]), (ex, h)
                        if valid[0, h]:
                            assert rew[0, h] == pytest.approx(r2) and tm[0, h] == t2
            state = nxt
            if term or trunc:
                break
    assert {colour_rm.NAV_NONE, colour_rm.NAV_WRONG, colour_rm.NAV_CORRECT, colour_rm.NAV_COMPLETE} <= set(seen)


def test_counterfactual_rules_reproduce_the_visited_transition():
    for key in ("delivery", "colour"):
        for env, step, info in rollouts(key, "oracle", 10):
            if step is None:
                continue
            rm, ri = env.rm, info["rm"]
            if ri["head"] < 0:
                continue
            ex = ri.get("extras", {})
            ex = {k: [ex.get(k, -1)] for k in ("cls", "w", "c_next")}
            valid, nh, rew, tm = rm.counterfactual(ex, [ri["head"]], [ri["next_head"]])
            h = ri["head"]
            assert valid[0, h] and rew[0, h] == pytest.approx(ri["reward"]) and tm[0, h] == ri["terminal"]
            assert nh[0, h] == ri["next_head"]
