"""Deferred Maintenance reward machine (protocol v2 section 4.2).

State: held code set H, stage index s, ready bit. Readiness is visible in the base
observation; H and s are not, so they are the task memory the RM supplies. PPO-RM sees
the factored encoding [multi-hot H (K), one-hot s (S), ready (1)].

The RM reproduces the env reward exactly on real trajectories (tested). It is used by an
on-policy agent, so it needs no counterfactual rules.
"""
import numpy as np


class MaintenanceRM:
    name = "maintenance"

    def __init__(self, max_code_types=3, max_stages=4, reward_service_station=100.0,
                 reward_perfect_completion=1000.0, penalty_premature_service=-10.0,
                 penalty_wrong_station=-10.0, penalty_off_path=-0.1,
                 penalty_leftover_codes=-1000.0, penalty_touch_boundary=-5.0):
        self.K = int(max_code_types)
        self.S = int(max_stages)
        self.r_service = reward_service_station
        self.r_complete = reward_perfect_completion
        self.p_premature = penalty_premature_service
        self.p_wrong = penalty_wrong_station
        self.p_off = penalty_off_path
        self.p_leftover = penalty_leftover_codes
        self.p_boundary = penalty_touch_boundary
        self.feature_size = self.K + self.S + 1
        self.terminal = None

    @classmethod
    def from_env(cls, env):
        core = env.unwrapped
        return cls(core.max_code_types, core.max_stages, core.reward_service_station,
                   core.reward_perfect_completion, core.penalty_premature_service,
                   core.penalty_wrong_station, core.penalty_off_path,
                   core.penalty_leftover_codes, core.penalty_touch_boundary)

    def initial_state(self, reset_events=()):
        return (tuple(), 0, 0)

    def step(self, state, events):
        if state is None:
            raise RuntimeError("step() called in the terminal state")
        held, stage, ready = list(state[0]), state[1], state[2]
        reward, terminal = 0.0, False
        for name, arg in events:
            if name == "code_collected":
                held.append(int(arg))
            elif name == "boundary_contact":
                reward += self.p_boundary
            elif name == "off_path":
                reward += self.p_off
            elif name == "service_attempt":
                k = int(arg)
                if ready and k in held:
                    reward += self.r_service
                    held.remove(k)
                elif not ready:
                    reward += self.p_premature
                else:
                    reward += self.p_wrong
            elif name == "readiness_on":
                ready = 1
            elif name == "stage_exit":
                stage += 1
            elif name == "stage_exit_incomplete":
                reward += self.p_leftover
                terminal = True
            elif name == "final_exit":
                reward += self.r_complete if not held else self.p_leftover
                terminal = True
            else:
                raise ValueError(f"unrecognised maintenance event {name!r}")
        nxt = None if terminal else (tuple(sorted(held)), stage, ready)
        return nxt, float(reward), terminal

    def features(self, state):
        f = np.zeros(self.feature_size, dtype=np.float32)
        if state is None:
            return f
        for k in state[0]:
            f[k - 1] = 1.0
        f[self.K + min(state[1], self.S - 1)] = 1.0
        f[self.K + self.S] = float(state[2])
        return f

    def head(self, state):
        return -1 if state is None else 0

    def describe(self, state):
        if state is None:
            return "terminal"
        return {"held": list(state[0]), "stage": state[1], "ready": state[2]}
