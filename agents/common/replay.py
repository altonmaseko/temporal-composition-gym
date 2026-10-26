"""Memory-bounded replay shared by DQN-family custom agents (DRQN, QRM).

Storage is a circular array of frames. Each step writes one frame, so a transition
(s_t, a_t, r_t, s_{t+1}) uses two consecutive slots and every observation is stored once.
An episode of T transitions occupies T + 1 slots (its final observation is kept so that
time-limit truncations can bootstrap from it).

Capacity unit: FRAMES. `capacity_frames` frames hold slightly fewer transitions than
frames (one extra frame per episode). Memory = capacity_frames x frame bytes (+ the
optional recurrent state per frame for DRQN burn-in).
"""
import numpy as np


class FrameReplay:
    def __init__(self, capacity_frames, obs_shape, obs_dtype, extras=None, hidden_shape=None):
        self.capacity = int(capacity_frames)
        n = self.capacity
        self.obs_shape = tuple(obs_shape)
        self.obs = np.zeros((n,) + self.obs_shape, dtype=obs_dtype)
        self.action = np.zeros(n, dtype=np.int64)
        self.reward = np.zeros(n, dtype=np.float32)
        self.terminated = np.zeros(n, dtype=bool)
        self.truncated = np.zeros(n, dtype=bool)
        self.has_transition = np.zeros(n, dtype=bool)
        self.episode = np.full(n, -1, dtype=np.int64)
        self.t = np.zeros(n, dtype=np.int64)
        self.extras = {}
        for name, (shape, dtype) in (extras or {}).items():
            self.extras[name] = np.zeros((n,) + tuple(shape), dtype=dtype)
        # Recurrent state *entering* the frame (the actor's state before reading it).
        self.hidden = None if hidden_shape is None else np.zeros((n,) + tuple(hidden_shape), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self._episode_id = -1
        self._cur_t = 0
        self._last = None
        self.num_transitions_added = 0

    @property
    def nbytes(self):
        total = self.obs.nbytes + self.action.nbytes + self.reward.nbytes + self.terminated.nbytes
        total += self.truncated.nbytes + self.has_transition.nbytes + self.episode.nbytes + self.t.nbytes
        total += sum(a.nbytes for a in self.extras.values())
        if self.hidden is not None:
            total += self.hidden.nbytes
        return total

    def _write_frame(self, obs, hidden):
        i = self.ptr
        self.obs[i] = obs
        self.episode[i] = self._episode_id
        self.t[i] = self._cur_t
        self.has_transition[i] = False
        if self.hidden is not None:
            self.hidden[i] = 0.0 if hidden is None else hidden
        self._last = i
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def start_episode(self, obs, hidden=None):
        """Write the first observation of a new episode."""
        self._episode_id += 1
        self._cur_t = 0
        self._write_frame(obs, hidden)

    def add(self, action, reward, terminated, truncated, next_obs, next_hidden=None, **extras):
        """Record the transition from the most recent frame to next_obs."""
        if self._last is None:
            raise RuntimeError("call start_episode() first")
        i = self._last
        self.action[i] = action
        self.reward[i] = reward
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        for name, value in extras.items():
            self.extras[name][i] = value
        self.has_transition[i] = True
        self._cur_t += 1
        self._write_frame(next_obs, next_hidden)
        self.num_transitions_added += 1

    # ------------------------------------------------------------------ validity
    def valid_transition(self, idx):
        """Vectorised: slot idx starts a transition whose next frame is still stored."""
        idx = np.asarray(idx) % self.capacity
        nxt = (idx + 1) % self.capacity
        return (self.has_transition[idx]
                & (self.episode[nxt] == self.episode[idx])
                & (self.t[nxt] == self.t[idx] + 1))

    def num_valid(self):
        return int(self.valid_transition(np.arange(self.size)).sum()) if self.size else 0

    def _sample_valid(self, n, rng):
        out = []
        for _ in range(1000):
            cand = rng.integers(0, self.size, size=max(4 * n, 16))
            out.extend(cand[self.valid_transition(cand)].tolist())
            if len(out) >= n:
                return np.array(out[:n], dtype=np.int64)
        raise RuntimeError("not enough valid transitions in replay")

    # ------------------------------------------------------------------ sampling
    def sample_transitions(self, batch_size, rng):
        idx = self._sample_valid(batch_size, rng)
        nxt = (idx + 1) % self.capacity
        batch = {
            "idx": idx,
            "obs": self.obs[idx],
            "next_obs": self.obs[nxt],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "terminated": self.terminated[idx],
            "truncated": self.truncated[idx],
        }
        for name, arr in self.extras.items():
            batch[name] = arr[idx]
        return batch

    def _same_episode_back(self, i, steps):
        """How many frames before slot i (up to `steps`) belong to the same episode."""
        k = 0
        while k < steps:
            j = (i - k - 1) % self.capacity
            if self.episode[j] != self.episode[i] or self.t[j] != self.t[i] - k - 1:
                break
            k += 1
        return k

    def sample_sequences(self, batch_size, train_len, burn_in, rng, episode_start_fraction=0.0):
        """Sample R2D2-style sequences that never cross an episode boundary.

        Each row holds up to `burn_in` frames of context before the first trained
        transition, then `train_len` transitions and their next frames. Rows start from
        the recurrent state stored with their first frame (zero at an episode start).
        A fraction of rows is aligned to the episode start (zero state, no burn-in), so
        long-range dependencies from the first frames are trained with exact state.

        Returns arrays of shape (B, T) with T = burn_in + train_len + 1 frames, and
        `loss_mask[b, k]` = 1 where position k is a trained, stored transition.
        """
        if self.hidden is None:
            raise RuntimeError("sequence sampling needs hidden_shape")
        T = burn_in + train_len + 1
        starts = self._sample_valid(batch_size, rng)
        obs = np.zeros((batch_size, T) + self.obs_shape, dtype=self.obs.dtype)
        action = np.zeros((batch_size, T), dtype=np.int64)
        reward = np.zeros((batch_size, T), dtype=np.float32)
        terminated = np.zeros((batch_size, T), dtype=bool)
        truncated = np.zeros((batch_size, T), dtype=bool)
        loss_mask = np.zeros((batch_size, T), dtype=np.float32)
        h0 = np.zeros((batch_size,) + self.hidden.shape[1:], dtype=np.float32)
        burn = np.zeros(batch_size, dtype=np.int64)
        for b, i in enumerate(starts):
            if episode_start_fraction > 0 and rng.random() < episode_start_fraction:
                back = self._same_episode_back(i, int(self.t[i]))
                if back == self.t[i]:          # whole episode prefix still stored
                    i = (i - back) % self.capacity
            k = self._same_episode_back(i, burn_in)
            first = (i - k) % self.capacity
            burn[b] = k
            h0[b] = self.hidden[first]
            pos = 0
            slot = first
            while pos < T:
                obs[b, pos] = self.obs[slot]
                if pos >= k and pos < k + train_len:
                    if not self.valid_transition(slot):
                        break
                    loss_mask[b, pos] = 1.0
                if self.has_transition[slot] and self.valid_transition(slot):
                    action[b, pos] = self.action[slot]
                    reward[b, pos] = self.reward[slot]
                    terminated[b, pos] = self.terminated[slot]
                    truncated[b, pos] = self.truncated[slot]
                    if self.terminated[slot] or self.truncated[slot]:
                        # write the final frame, then stop (never cross into the next episode)
                        if pos + 1 < T:
                            obs[b, pos + 1] = self.obs[(slot + 1) % self.capacity]
                        break
                    pos += 1
                    slot = (slot + 1) % self.capacity
                else:
                    break
        return {
            "obs": obs, "action": action, "reward": reward, "terminated": terminated,
            "truncated": truncated, "loss_mask": loss_mask, "h0": h0, "burn_in": burn,
        }
