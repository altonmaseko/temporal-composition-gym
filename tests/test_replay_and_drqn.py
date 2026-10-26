"""FrameReplay and DRQN memory mechanics (protocol v2 section 6.2)."""
import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.common.env_factory import make_env
from agents.common.replay import FrameReplay
from agents.common.seeding import RunRNG
from agents.drqn.drqn_core import DRQN_DEFAULTS, DRQNAgent, DRQNConfig, train_drqn


def fill(replay, n_episodes, rng):
    """Synthetic episodes: obs = [episode, t]; stored hidden = episode*1000 + t."""
    for ep in range(n_episodes):
        length = int(rng.integers(1, 15))
        replay.start_episode(np.array([ep, 0], dtype=np.float32), np.full((2, 1), ep * 1000.0))
        for t in range(length):
            end = t == length - 1
            term = end and ep % 2 == 0
            replay.add(t % 3, float(t), term, end and not term, np.array([ep, t + 1], dtype=np.float32),
                       next_hidden=np.full((2, 1), ep * 1000.0 + t + 1))


def test_transitions_survive_wraparound():
    rng = np.random.default_rng(0)
    rb = FrameReplay(64, (2,), np.float32, hidden_shape=(2, 1))
    fill(rb, 40, rng)
    b = rb.sample_transitions(256, rng)
    assert np.all(b["next_obs"][:, 0] == b["obs"][:, 0])
    assert np.all(b["next_obs"][:, 1] == b["obs"][:, 1] + 1)


@pytest.mark.parametrize("start_fraction", [0.0, 1.0])
def test_sequences_never_cross_episodes_and_start_from_stored_state(start_fraction):
    rng = np.random.default_rng(1)
    rb = FrameReplay(200, (2,), np.float32, hidden_shape=(2, 1))
    fill(rb, 60, rng)
    s = rb.sample_sequences(64, train_len=5, burn_in=3, rng=rng, episode_start_fraction=start_fraction)
    for b in range(64):
        ep, t0 = s["obs"][b, 0]
        assert s["h0"][b, 0, 0] == ep * 1000 + t0                 # actor state stored with frame 0
        assert s["burn_in"][b] <= 3
        trained = np.flatnonzero(s["loss_mask"][b])
        assert len(trained) > 0 and trained[0] == s["burn_in"][b]
        for pos in trained:                                         # trained step and its next frame
            assert tuple(s["obs"][b, pos]) == (ep, t0 + pos)
            assert tuple(s["obs"][b, pos + 1]) == (ep, t0 + pos + 1)
        last = trained[-1]
        assert not np.any(s["loss_mask"][b, last + 1:])
        if s["terminated"][b, last] or s["truncated"][b, last]:
            assert np.all(s["obs"][b, last + 2:] == 0)             # padding, never the next episode
    if start_fraction == 1.0:
        assert np.mean(s["obs"][:, 0, 1] == 0) > 0.5               # mostly aligned to episode starts


def test_image_replay_memory_is_bounded():
    cfg = DRQN_DEFAULTS["colour"]
    rb = FrameReplay(1000, (84, 84, 3), np.uint8, hidden_shape=(2, cfg.lstm_dim))
    per_frame = rb.nbytes / 1000
    assert 84 * 84 * 3 <= per_frame < 84 * 84 * 3 + 2 * cfg.lstm_dim * 4 + 64
    assert per_frame * cfg.replay_frames < 2.5e9                     # protocol estimate ~2.1 GB + state


def tiny_agent(obs_space, n_actions, **kw):
    cfg = DRQNConfig(enc_dim=32, lstm_dim=32, replay_frames=5000, **kw)
    return DRQNAgent(obs_space, n_actions, cfg, "cpu", RunRNG(0))


def test_hidden_state_advances_on_random_actions():
    env = make_env("delivery")
    agent = tiny_agent(env.observation_space, 4)
    obs, _ = env.reset(seed=0)
    agent.begin_episode(obs)
    agent.act(obs, epsilon=1.0)
    assert float(agent.hidden[0].abs().sum()) > 0
    stored = agent.replay.hidden[agent.replay._last]
    assert np.all(stored == 0)                                        # episode start state is zero


def test_truncation_bootstraps_and_termination_does_not():
    space = gym.spaces.Box(-1, 1, (3,), np.float32)
    agent = tiny_agent(space, 2)
    batch = {
        "obs": np.random.default_rng(0).normal(size=(2, 3, 3)).astype(np.float32),
        "action": np.zeros((2, 3), np.int64), "reward": np.ones((2, 3), np.float32),
        "terminated": np.array([[1, 0, 0], [0, 0, 0]], bool), "truncated": np.array([[0, 0, 0], [1, 0, 0]], bool),
        "loss_mask": np.array([[1, 0, 0], [1, 0, 0]], np.float32), "h0": np.zeros((2, 2, 32), np.float32),
    }
    _, y, m = agent.targets(batch)
    with torch.no_grad():
        z = torch.zeros(1, 2, 32)
        qt, _ = agent.target(torch.as_tensor(batch["obs"]), (z, z))
    assert float(y[0, 0]) == pytest.approx(1.0)
    assert float(y[1, 0]) == pytest.approx(1.0 + 0.99 * float(qt[1, 1].max()))
    assert m.tolist() == [[1, 0], [1, 0]]


class CueRecall(gym.Env):
    """A cue (+1/-1) at t=0, `delay` blank steps, then a decision: reward +1 if the action
    matches the cue. A memoryless policy can only reach 50%."""
    observation_space = gym.spaces.Box(-1, 1, (3,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, delay=4):
        self.delay = delay

    def _obs(self):
        if self.t == 0:
            return np.array([2 * self.cue - 1, 1, 0], np.float32)
        if self.t == self.delay + 1:
            return np.array([0, 0, 1], np.float32)
        return np.zeros(3, np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cue, self.t = int(self.np_random.integers(2)), 0
        return self._obs(), {}

    def step(self, a):
        if self.t == self.delay + 1:
            return self._obs(), (1.0 if a == self.cue else -1.0), True, False, {"is_success": a == self.cue}
        self.t += 1
        return self._obs(), 0.0, False, False, {}


@pytest.mark.slow
def test_drqn_learns_a_delayed_recall_task():
    torch.manual_seed(0)
    env = CueRecall(delay=4)
    agent = tiny_agent(env.observation_space, 2, lr=3e-3, learning_starts=300, train_freq=1,
                       target_update=100, batch_size=16, train_len=8, burn_in=4, eps_fraction=0.3)
    train_drqn(env, agent, 5000, reset_seed=0)
    wins = 0
    for ep in range(200):
        obs, _ = env.reset(seed=10_000 + ep)
        hidden = agent.zero_hidden()
        while True:
            q, hidden = agent.policy_step(obs, hidden)
            obs, r, term, _, _ = env.step(int(np.argmax(q)))
            if term:
                wins += r > 0
                break
    assert wins / 200 >= 0.9


def test_drqn_colour_smoke_runs_updates_on_rgb_frames():
    env = make_env("colour")
    cfg = DRQNConfig(encoder="cnn", enc_dim=64, lstm_dim=32, replay_frames=2000, learning_starts=200,
                     batch_size=2, train_len=16, burn_in=8, episode_start_fraction=0.5)
    agent = DRQNAgent(env.observation_space, 4, cfg, "cpu", RunRNG(0))
    losses = []
    train_drqn(env, agent, 400, reset_seed=0, on_update=lambda s, e, m: losses.append(m["loss"]))
    assert agent.replay.obs.dtype == np.uint8 and agent.replay.obs.shape[-1] == 3
    assert agent.n_updates == len(losses) == (400 - 200) // 4 + 1 and np.all(np.isfinite(losses))
