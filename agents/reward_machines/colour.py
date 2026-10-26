"""Cue-recording reward machine for Sequential Colour, used by QRM-Cue (protocol v2 section 4.3).

RM state u = (phase, R, k):
  phase  'cue' | 'delay' | 'nav'  (terminal state = None)
  R      colour ids recorded from `cue` events, in order (repeats kept)
  k      walls completed so far
It is built only from info["events"]: `cue` events arrive while the cue is on screen and
`wall_touch` events carry only the touched wall's colour. The env's hidden sequence and
progress index are never read.

Q heads use the abstraction phi(u):
  cue phase,   j cues recorded           -> head j-1                      (L heads)
  delay phase, j cues recorded           -> head L + j-1                  (L heads)
  navigation, target c = R[k], r = |R|-k -> head 2L + c*L + (r-1)         (V*L heads)
Every room is re-randomised independently and the agent re-centred after each correct
wall, so from a navigation state the optimal behaviour and value depend only on the
current target and how many walls remain, not on later targets. That symmetry needs
randomised wall colours, a vocabulary of at least 4 and finite episodes; from_env checks it.

Counterfactual updates are applied only where the env would have behaved identically:
  navigation step, no wall touched -> every navigation head, same head next
  wrong wall w (agent re-centred)  -> navigation heads with target != w
  correct wall w, episode goes on  -> heads (w, r>=2), next head (actual next target, r-1)
  sequence completed on wall w     -> head (w, 1) only
  cue / delay steps                -> the actual head only (cue content is observed)
"""
import numpy as np

PRE, NAV_NONE, NAV_WRONG, NAV_CORRECT, NAV_COMPLETE = range(5)


class CueRecordingRM:
    name = "colour"

    def __init__(self, vocab_size=6, max_len=5, reward_density="sparse", reward_correct_wall=1.0,
                 reward_sequence_completion=10.0, penalty_severity=1.0, hard_death_mode=False):
        self.V = int(vocab_size)
        self.L = int(max_len)
        self.dense = reward_density == "dense"
        self.r_correct = float(reward_correct_wall)
        self.r_complete = float(reward_sequence_completion)
        self.penalty = float(penalty_severity)
        self.hard_death = bool(hard_death_mode)
        self.n_states = 2 * self.L + self.V * self.L       # number of Q heads
        self.feature_size = self.n_states
        self.terminal = None
        heads = np.arange(self.n_states)
        self.is_nav = heads >= 2 * self.L
        nav_off = np.where(self.is_nav, heads - 2 * self.L, 0)
        self.head_c = np.where(self.is_nav, nav_off // self.L, -1)
        self.head_r = np.where(self.is_nav, nav_off % self.L + 1, 0)

    @classmethod
    def from_env(cls, env):
        core = env.unwrapped
        if not core.randomise_wall_colors or core.colour_vocabulary_size < 4 or core.infinite_mode:
            raise ValueError(
                "QRM-Cue's head abstraction assumes randomised wall colours, a vocabulary of at "
                "least 4 colours and finite episodes; re-validate it before using other settings")
        return cls(core.colour_vocabulary_size, core.sequence_length_range[1], core.reward_density,
                   core.reward_correct_wall, core.reward_sequence_completion,
                   core.penalty_severity, core.hard_death_mode)

    # -- heads
    def nav_head(self, c, r):
        return 2 * self.L + int(c) * self.L + (int(r) - 1)

    def head(self, state):
        if state is None:
            return -1
        phase, R, k = state
        j = len(R)
        if phase == "cue":
            return j - 1
        if phase == "delay":
            return self.L + j - 1
        return self.nav_head(R[k], j - k)

    def features(self, state):
        f = np.zeros(self.feature_size, dtype=np.float32)
        h = self.head(state)
        if h >= 0:
            f[h] = 1.0
        return f

    def describe(self, state):
        if state is None:
            return "terminal"
        return {"phase": state[0], "recorded": list(state[1]), "walls_done": state[2]}

    # -- tracking
    def _apply(self, state, events):
        phase, R, k = state
        R = list(R)
        reward, terminal = 0.0, False
        for name, arg in events:
            if name == "cue":
                if phase != "cue":
                    raise RuntimeError("cue event outside the cue phase")
                R.append(int(arg))
                if len(R) > self.L:
                    raise RuntimeError("more cues than the configured maximum length")
            elif name == "delay_start":
                phase = "delay"
            elif name == "nav_start":
                phase, k = "nav", 0
            elif name == "wall_touch":
                if phase != "nav":
                    raise RuntimeError("wall_touch outside navigation")
                if int(arg) == R[k]:
                    if self.dense:
                        reward += self.r_correct
                    k += 1
                    if k == len(R):
                        if not self.dense:
                            reward += self.r_complete
                        terminal = True
                else:
                    reward -= self.penalty
                    if self.hard_death:
                        terminal = True
            elif name == "room_reset":
                pass
            else:
                raise ValueError(f"unrecognised colour event {name!r}")
        return (None if terminal else (phase, tuple(R), k)), reward, terminal

    def initial_state(self, reset_events):
        state, _, _ = self._apply(("cue", tuple(), 0), reset_events)
        return state

    def step(self, state, events):
        if state is None:
            raise RuntimeError("step() called in the terminal state")
        nxt, reward, terminal = self._apply(state, events)
        return nxt, float(reward), terminal

    # -- QRM
    def transition_extras(self, state, events, next_state):
        phase, R, k = state
        touches = [int(a) for n, a in events if n == "wall_touch"]
        if phase != "nav":
            return {"cls": PRE, "w": -1, "c_next": -1}
        if not touches:
            return {"cls": NAV_NONE, "w": -1, "c_next": -1}
        w = touches[0]
        if w != R[k]:
            return {"cls": NAV_WRONG, "w": w, "c_next": -1}
        if k + 1 == len(R):
            return {"cls": NAV_COMPLETE, "w": w, "c_next": -1}
        return {"cls": NAV_CORRECT, "w": w, "c_next": R[k + 1]}

    def counterfactual(self, extras, actual_head, actual_next_head):
        """(valid, next_head, reward, terminal), each (B, n_heads), for a replay batch."""
        cls = np.asarray(extras["cls"], dtype=np.int64)[:, None]
        w = np.asarray(extras["w"], dtype=np.int64)[:, None]
        c_next = np.asarray(extras["c_next"], dtype=np.int64)[:, None]
        actual_head = np.asarray(actual_head, dtype=np.int64)[:, None]
        actual_next = np.asarray(actual_next_head, dtype=np.int64)[:, None]
        B, H = cls.shape[0], self.n_states
        heads = np.broadcast_to(np.arange(H), (B, H))
        nav = np.broadcast_to(self.is_nav, (B, H))
        hc = np.broadcast_to(self.head_c, (B, H))
        hr = np.broadcast_to(self.head_r, (B, H))

        valid = np.zeros((B, H), dtype=bool)
        nxt = heads.copy()
        rew = np.zeros((B, H), dtype=np.float32)
        term = np.zeros((B, H), dtype=bool)

        pre = cls == PRE
        valid |= pre & (heads == actual_head)
        nxt = np.where(pre, actual_next, nxt)

        valid |= (cls == NAV_NONE) & nav

        wrong = cls == NAV_WRONG
        m = wrong & nav & (hc != w)
        valid |= m
        rew = np.where(m, -self.penalty, rew)
        term = np.where(m, self.hard_death, term)

        corr = cls == NAV_CORRECT
        m = corr & nav & (hc == w) & (hr >= 2)
        valid |= m
        nxt = np.where(m, 2 * self.L + c_next * self.L + (hr - 2), nxt)
        rew = np.where(m, self.r_correct if self.dense else 0.0, rew)

        comp = cls == NAV_COMPLETE
        m = comp & nav & (hc == w) & (hr == 1)
        valid |= m
        rew = np.where(m, self.r_correct if self.dense else self.r_complete, rew)
        term = np.where(m, True, term)

        nxt = np.where(term, -1, nxt)
        return valid, nxt, rew.astype(np.float32), term
