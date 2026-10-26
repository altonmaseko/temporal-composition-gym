"""QRM learner: vectorised counterfactual targets, truncation, learning from intended transitions."""
import numpy as np
import pytest
import torch

from agents.common.env_factory import make_env
from agents.common.seeding import RunRNG
from agents.qrm.qrm_core import QRMAgent, QRMConfig, train_qrm
from agents.reward_machines import RewardMachineWrapper, make_reward_machine
from agents.reward_machines import delivery as drm


def agent_for(key, **kw):
    base = make_env(key)
    rm = make_reward_machine(key, base)
    env = RewardMachineWrapper(base, rm, reward_mode="rm")
    enc = "cnn" if key == "colour" else "mlp"
    cfg = QRMConfig(encoder=enc, enc_dim=64, hidden_dim=32, replay_frames=3000, **kw)
    return env, rm, QRMAgent(env.observation_space, 4, rm, cfg, "cpu", RunRNG(0))


@pytest.mark.parametrize("key", ["delivery", "colour"])
def test_vectorised_targets_match_a_per_head_loop(key):
    torch.manual_seed(0)
    env, rm, agent = agent_for(key, learning_starts=10 ** 9)
    train_qrm(env, agent, 1500, reset_seed=0)
    batch = agent.replay.sample_transitions(16, np.random.default_rng(0))
    q_sa, y, valid = agent.targets(batch)
    v, nh, rw, tm = rm.counterfactual({k: batch[k] for k in agent.EXTRA_KEYS}, batch["head"], batch["next_head"])
    with torch.no_grad():
        q_all = agent.q(torch.as_tensor(batch["obs"]))
        qt_all = agent.target(torch.as_tensor(batch["next_obs"]))
    for b in range(16):
        assert v[b, batch["head"][b]], "visited head is always updated"
        for h in range(rm.n_states):
            assert bool(valid[b, h]) == bool(v[b, h])
            if not v[b, h]:
                continue
            boot = 0.0 if tm[b, h] else 0.99 * float(qt_all[b, nh[b, h]].max())
            assert float(y[b, h]) == pytest.approx(rw[b, h] + boot, abs=1e-5)
            assert float(q_sa[b, h]) == pytest.approx(float(q_all[b, h, batch["action"][b]]), abs=1e-6)


def test_truncation_bootstraps_but_rm_terminal_does_not():
    env, rm, agent = agent_for("delivery")
    obs = np.zeros((2, 8), np.float32)
    batch = {"obs": obs, "next_obs": obs + 1, "action": np.zeros(2, np.int64),
             "terminated": np.array([False, True]), "truncated": np.array([True, False]),
             "head": np.array([0, 0]), "next_head": np.array([0, -1]),
             "cls": np.array([drm.NONE, drm.LAVA]), "w": np.array([-1, -1]), "c_next": np.array([-1, -1])}
    _, y, valid = agent.targets(batch)
    with torch.no_grad():
        boot = agent.target(torch.as_tensor(obs + 1))[0].max(-1).values
    assert torch.allclose(y[0], 0.99 * boot, atol=1e-6) and bool(valid[0].all())      # truncated: bootstrap
    assert torch.allclose(y[1], torch.full_like(y[1], -50.0)) and bool(valid[1].all())  # lava: terminal


def test_replay_stores_rm_rewards_heads_and_event_classes():
    env, rm, agent = agent_for("delivery", learning_starts=10 ** 9)
    train_qrm(env, agent, 800, reset_seed=3)
    rb = agent.replay
    idx = np.flatnonzero(rb.valid_transition(np.arange(rb.size)))
    assert set(np.unique(rb.reward[idx])) <= {0.0, -1.0, -51.0, -50.0, 10.0, 50.0}
    assert np.all(rb.extras["head"][idx] >= 0)
    assert np.all((rb.extras["cls"][idx] >= 0) & (rb.extras["cls"][idx] < drm.N_CLASSES))


@pytest.mark.parametrize("key", ["delivery", "colour"])
def test_qrm_updates_use_counterfactual_heads(key):
    env, rm, agent = agent_for(key, learning_starts=200, batch_size=8)
    stats = []
    train_qrm(env, agent, 600, reset_seed=0, on_update=lambda s, e, m: stats.append(m))
    assert agent.n_updates == len(stats) == (600 - 200) // 4 + 1
    assert all(np.isfinite(m["loss"]) for m in stats)
    assert np.mean([m["valid_heads_per_transition"] for m in stats]) > 1.0
