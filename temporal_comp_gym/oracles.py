"""Scripted privileged controllers ("oracles") for the three tasks.

They read hidden environment state directly and exist only to show that each task is
solvable within its episode limit, independently of any learning agent, and to give an
upper reference score. They are never used to train or evaluate learning agents.
"""
import heapq
import math
from collections import deque

import numpy as np


def _core(env):
    return env.unwrapped


# --------------------------------------------------------------------------- delivery
class DeliveryOracle:
    """Dijkstra planner over the full grid: lava is impassable, damage is expensive."""

    MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    def __init__(self, env, damage_cost=25.0):
        self.env = env
        self.damage_cost = damage_cost

    def reset(self):
        pass

    def _cost(self, cell, allow_goal):
        if cell is None:
            return 1.0
        if cell.type in ("wall", "lava"):
            return None
        if cell.type == "goal" and not allow_goal:
            return None
        if cell.type == "damage":
            return self.damage_cost
        return 1.0

    def _first_action(self, targets, allow_goal):
        core = _core(self.env)
        start = tuple(int(v) for v in core.agent_pos)
        targets = set(targets)
        dist = {start: 0.0}
        first = {start: None}
        heap = [(0.0, 0, start)]
        tie = 0
        while heap:
            d, _, pos = heapq.heappop(heap)
            if d > dist[pos]:
                continue
            if pos in targets and pos != start:
                return first[pos]
            for a, (dx, dy) in self.MOVES.items():
                nxt = (pos[0] + dx, pos[1] + dy)
                cell = core.grid.get(*nxt)
                is_target = nxt in targets
                c = self._cost(cell, allow_goal or is_target)
                if c is None:
                    continue
                nd = d + c
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    first[nxt] = a if first[pos] is None else first[pos]
                    tie += 1
                    heapq.heappush(heap, (nd, tie, nxt))
                    # goal and objectives are endpoints, not waypoints
        return None

    def act(self):
        core = _core(self.env)
        kinds = {"package": [], "destination": [], "goal": []}
        for i in range(core.grid.width):
            for j in range(core.grid.height):
                cell = core.grid.get(i, j)
                if cell is not None and cell.type in kinds:
                    kinds[cell.type].append((i, j))
        if core.delivered_packages >= core.num_packages:
            targets, allow_goal = kinds["goal"], True
        elif core.carried_packages > 0 and (
                core.carried_packages >= core.carrying_capacity or not kinds["package"]):
            targets, allow_goal = kinds["destination"], False
        else:
            targets, allow_goal = kinds["package"], False
        a = self._first_action(targets, allow_goal)
        return 0 if a is None else a


# --------------------------------------------------------------------------- colour
class ColourOracle:
    """Moves straight towards the wall showing the env's current target colour."""

    def __init__(self, env):
        self.env = env

    def reset(self):
        pass

    def act(self):
        core = _core(self.env)
        if core.current_phase != "navigation":
            return 0
        target = core.sequence[core.current_seq_idx]
        side = next(k for k, v in core.wall_colors.items() if v == target)
        return {"top": 0, "bottom": 1, "left": 2, "right": 3}[side]


# --------------------------------------------------------------------------- maintenance
class MaintenanceOracle:
    """Grid planner + PD flight controller with access to the episode schedule.

    Plan per stage: collect every code zone, then (once ready) hover on a station for each
    held code until it is serviced, then fly to the exit. Paths avoid the exit cell until
    the stage is finished and avoid stations of other types.
    """

    def __init__(self, env, kp=6.0, kd=4.0, max_acc=700.0, max_tilt=0.7):
        self.env = env
        self.kp, self.kd, self.max_acc, self.max_tilt = kp, kd, max_acc, max_tilt
        self._plan_key = None
        self._path = []

    def reset(self):
        self._plan_key = None
        self._path = []

    # -- targets
    def _goal(self, core):
        from temporal_comp_gym.envs.deferred_maintenance.objects import CodeZone, MaintenanceStation
        pos = np.array(core.rocket.body.position)
        codes = [e for e in core.entities if isinstance(e, CodeZone)]
        if codes:
            c = min(codes, key=lambda e: np.sum((np.array(e.body.position) - pos) ** 2))
            return core._cell_of(c.body.position), "pass"
        if core.readiness == 1.0 and core.held_codes:
            k = core.held_codes[0]
            st = [e for e in core.entities if isinstance(e, MaintenanceStation) and e.color_code == k]
            s = min(st, key=lambda e: np.sum((np.array(e.body.position) - pos) ** 2))
            return core._cell_of(s.body.position), "hover"
        return tuple(core.path_cells[-1]), "exit"

    def _plan(self, core, start, goal, mode):
        from temporal_comp_gym.envs.deferred_maintenance.core import CELL_BOUNDARY, CELL_EXIT, CELL_STATION
        blocked = set()
        for cell, t in core.cell_types.items():
            if t == CELL_BOUNDARY or (t == CELL_EXIT and mode != "exit"):
                blocked.add(cell)
            if t == CELL_STATION and cell != goal:
                blocked.add(cell)
        blocked.discard(goal)
        if mode != "exit":
            # Keep a one-cell margin around the exit when possible: the rocket body is
            # taller than the gap and can clip the exit tile while passing next to it.
            exit_c = tuple(core.path_cells[-1])
            margin = {(exit_c[0] + dx, exit_c[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            margin.discard(goal)
            path = self._bfs(core, start, goal, blocked | margin)
            if path is not None:
                return path
        return self._bfs(core, start, goal, blocked) or [goal]

    @staticmethod
    def _bfs(core, start, goal, blocked):
        prev = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + dx, c[1] + dy)
                if n in prev or n in blocked or n not in core.cell_types:
                    continue
                prev[n] = c
                q.append(n)
        if goal not in prev:
            return None
        path = []
        c = goal
        while c is not None:
            path.append(c)
            c = prev[c]
        return path[::-1][1:] or [goal]

    def act(self):
        core = _core(self.env)
        body = core.rocket.body
        pos = np.array(body.position, dtype=float)
        vel = np.array(body.velocity, dtype=float)
        goal, mode = self._goal(core)
        key = (core.stages_traversed, goal, mode, len(core.held_codes))
        cell = core._cell_of(pos)
        if key != self._plan_key:
            self._plan_key = key
            self._path = self._plan(core, cell, goal, mode)
        # drop waypoints already reached
        while len(self._path) > 1:
            c = self._path[0]
            centre = (np.array(c) + 0.5) * core.cell_size
            if np.linalg.norm(centre - pos) < 18.0:
                self._path.pop(0)
            else:
                break
        target = (np.array(self._path[0]) + 0.5) * core.cell_size
        final = len(self._path) == 1
        kp, kd = (self.kp, self.kd) if final else (self.kp * 0.8, self.kd * 0.7)
        acc = kp * (target - pos) - kd * vel
        n = np.linalg.norm(acc)
        if n > self.max_acc:
            acc *= self.max_acc / n
        g = core.gravity_scale
        tx, ty = acc[0], acc[1] - g          # required thrust vector (screen y points down)
        if ty >= 0:                           # need to fall faster than gravity: no thrust
            return np.array([0.0, self._torque(body, 0.0)], dtype=np.float32)
        theta_star = float(np.clip(math.atan2(tx, -ty), -self.max_tilt, self.max_tilt))
        mag = math.hypot(tx, ty) * body.mass
        err = theta_star - body.angle
        thrust = float(np.clip(mag / 2000.0 * max(math.cos(err), 0.0), 0.0, 1.0))
        return np.array([thrust, self._torque(body, theta_star)], dtype=np.float32)

    @staticmethod
    def _torque(body, theta_star):
        cmd = 3.0 * (theta_star - body.angle) - 0.08 * body.angular_velocity
        if abs(theta_star) < 0.03 and abs(cmd) < 0.06:
            return 0.0                        # let the env's restoring torque level the rocket
        if 0 < abs(cmd) < 0.06:
            cmd = math.copysign(0.06, cmd)
        return float(np.clip(cmd, -1.0, 1.0))


ORACLES = {
    "delivery": DeliveryOracle,
    "colour": ColourOracle,
    "maintenance": MaintenanceOracle,
}


def run_oracle_episode(env, oracle, seed, max_steps=None):
    """Roll out one oracle episode; returns a summary dict (used by tests and reports)."""
    obs, info = env.reset(seed=seed)
    oracle.reset()
    total, steps = 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(oracle.act())
        total += r
        steps += 1
        if term or trunc or (max_steps is not None and steps >= max_steps):
            return {
                "seed": seed, "return": total, "length": steps,
                "success": bool(info.get("is_success", False)),
                "cause": info.get("termination_cause"),
            }
