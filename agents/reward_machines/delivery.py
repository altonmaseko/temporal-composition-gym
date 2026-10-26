"""Hazardous Delivery reward machine (protocol v2 section 4.1).

State (d, c): deliveries so far and packages carried, d + c <= N, c <= capacity, plus an
absorbing terminal state. The labelling function is the env's info["events"].

Counterfactual validity. The env's hidden counters mirror (d, c), so its response on
package and destination squares depends on them. A counterfactual state u' gets an update
from a transition only if the env would have produced the same transition in u':
  pickup               -> c' < cap and d' + c' < N  (a package still exists)
  package contact only -> c' >= cap and d' + c' < N
  delivery             -> c' > 0
  destination only     -> c' == 0 and d' < N         (a destination still exists)
  everything else      -> every state
"""
import numpy as np

# Event classes: one per step (each step has at most one cell effect).
NONE, DAMAGE, DAMAGE_DEATH, HEAL, LAVA, PKG_CONTACT, PICKUP, DEST_CONTACT, DELIVERY, GOAL = range(10)
CLASS_NAMES = ["none", "damage", "damage_death", "heal", "lava", "package_contact",
               "pickup", "destination_contact", "delivery", "goal_contact"]
N_CLASSES = len(CLASS_NAMES)


def classify(events):
    names = {e[0] for e in events}
    if not names:
        return NONE
    if "lava" in names:
        return LAVA
    if "health_depleted" in names:
        return DAMAGE_DEATH
    if "damage" in names:
        return DAMAGE
    if "heal" in names:
        return HEAL
    if "goal_contact" in names:
        return GOAL
    if "delivery" in names:
        return DELIVERY
    if "destination_contact" in names:
        return DEST_CONTACT
    if "pickup" in names:
        return PICKUP
    if "package_contact" in names:
        return PKG_CONTACT
    raise ValueError(f"unrecognised delivery events {events}")


class DeliveryRM:
    name = "delivery"

    def __init__(self, num_packages=3, carrying_capacity=1, reward_delivery=10.0,
                 reward_task_completion=50.0, penalty_damage=-1.0,
                 penalty_premature_goal=-50.0, penalty_death_square=-50.0):
        self.N = int(num_packages)
        self.cap = int(carrying_capacity)
        self.states = [(d, c) for d in range(self.N + 1) for c in range(self.cap + 1) if d + c <= self.N]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.n_states = len(self.states)          # live states (Q heads)
        self.terminal = self.n_states              # absorbing terminal index
        self.feature_size = self.n_states
        H, C = self.n_states, N_CLASSES
        self.next = np.tile(np.arange(H)[:, None], (1, C))
        self.reward = np.zeros((H, C), dtype=np.float32)
        self.is_terminal = np.zeros((H, C), dtype=bool)
        self.valid = np.zeros((H, C), dtype=bool)
        for u, (d, c) in enumerate(self.states):
            for cls in range(C):
                nxt, r, term, ok = (d, c), 0.0, False, True
                if cls == DAMAGE:
                    r = penalty_damage
                elif cls == DAMAGE_DEATH:
                    r, term = penalty_damage + penalty_death_square, True
                elif cls == LAVA:
                    r, term = penalty_death_square, True
                elif cls == GOAL:
                    r = reward_task_completion if d == self.N else penalty_premature_goal
                    term = True
                elif cls == PICKUP:
                    ok = c < self.cap and d + c < self.N
                    if ok:
                        nxt = (d, c + 1)
                elif cls == PKG_CONTACT:
                    ok = c >= self.cap and d + c < self.N
                elif cls == DELIVERY:
                    ok = c > 0
                    if ok:
                        nxt, r = (d + 1, c - 1), reward_delivery
                elif cls == DEST_CONTACT:
                    ok = c == 0 and d < self.N
                self.next[u, cls] = self.terminal if term else self.index[nxt]
                self.reward[u, cls] = r
                self.is_terminal[u, cls] = term
                self.valid[u, cls] = ok

    @classmethod
    def from_env(cls, env):
        """Build from the env's fixed configuration (not its hidden state)."""
        core = env.unwrapped
        return cls(core.num_packages, core.carrying_capacity, core.reward_delivery,
                   core.reward_task_completion, core.penalty_damage,
                   core.penalty_premature_goal, core.penalty_death_square)

    # -- tracking
    def initial_state(self, reset_events=()):
        return self.index[(0, 0)]

    def step(self, u, events):
        if u == self.terminal:
            raise RuntimeError("step() called in the terminal state")
        cls = classify(events)
        if not self.valid[u, cls]:
            raise RuntimeError(f"event {CLASS_NAMES[cls]} impossible in RM state {self.states[u]}")
        return int(self.next[u, cls]), float(self.reward[u, cls]), bool(self.is_terminal[u, cls])

    def head(self, u):
        return -1 if u == self.terminal else u

    def features(self, u):
        f = np.zeros(self.feature_size, dtype=np.float32)
        if u != self.terminal:
            f[u] = 1.0
        return f

    def describe(self, u):
        return "terminal" if u == self.terminal else {"delivered": self.states[u][0], "carried": self.states[u][1]}

    # -- QRM
    def transition_extras(self, u, events, u_next):
        return {"cls": classify(events)}

    def counterfactual(self, extras, actual_head, actual_next_head):
        """(valid, next_head, reward, terminal), each (B, n_states), for a replay batch."""
        cls = np.asarray(extras["cls"], dtype=np.int64)
        valid = self.valid[:, cls].T
        nxt = self.next[:, cls].T
        rew = self.reward[:, cls].T
        term = self.is_terminal[:, cls].T
        nxt = np.where(term, -1, nxt)
        return valid, nxt, rew, term
