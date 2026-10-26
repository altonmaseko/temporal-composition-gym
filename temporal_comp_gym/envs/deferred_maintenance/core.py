import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
import requests
import io
import math
from .objects import Rocket, PathBoundary, CodeZone, MaintenanceStation, StageExit, StageEntry, StageBoundary

# URL Override System for rendering icons
ICON_URLS = {
    "rocket": "",
    "code_zone": "",
    "station": "",
    "stage_start": "",
    "stage_exit": ""
}

# 12 Distinct Colors for Visualization
VOCAB_COLORS_12 = [
    "#FF4444", # Red
    "#44FF44", # Green
    "#4444FF", # Blue
    "#FFFF44", # Yellow
    "#FF44FF", # Magenta
    "#44FFFF", # Cyan
    "#FFA500", # Orange
    "#800080", # Purple
    "#FFC0CB", # Pink
    "#32CD32", # Lime
    "#008080", # Teal
    "#A52A2A"  # Brown
]

def get_color_hex(type_id):
    return VOCAB_COLORS_12[(type_id - 1) % 12]

# Cell type codes. They equal the pymunk collision types in objects.py and are what the
# local-grid sensor reports.
CELL_VOID = 0
CELL_PATH = 2
CELL_CODE = 3
CELL_STATION = 4
CELL_EXIT = 5
CELL_ENTRY = 6
CELL_BOUNDARY = 7
SAFE_CELLS = {CELL_PATH, CELL_CODE, CELL_STATION, CELL_EXIT, CELL_ENTRY, CELL_BOUNDARY}

# Rocket shapes get their own query category so sensor queries never return the rocket.
ROCKET_CATEGORY = 0b10
SENSOR_FILTER = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ ROCKET_CATEGORY)

TERMINATION_CAUSES = ("success", "uncollected_codes_at_exit", "held_codes_at_final_exit", "timeout")

# Hand-authored CUSTOM_STAGES were removed in protocol v2: every stage is generated from the
# seeded episode schedule so that each episode is solvable by construction.

_IMAGE_CACHE = {}

def get_icon(entity_type, color_hex="#FFFFFF", size=32):
    cache_key = f"{entity_type}_{color_hex}_{size}"
    if cache_key in _IMAGE_CACHE:
        return _IMAGE_CACHE[cache_key]

    url = ICON_URLS.get(entity_type, "")
    surf = None
    if url:
        try:
            resp = requests.get(url, timeout=1.0)
            if resp.status_code == 200:
                surf = pygame.image.load(io.BytesIO(resp.content)).convert_alpha()
                surf = pygame.transform.scale(surf, (size, size))
        except Exception:
            pass 

    if surf is None:
        from pytablericons import TablerIcons
        from pytablericons.outline_icon import OutlineIcon
        
        pil_img = None
        if entity_type == "rocket":
            pil_img = TablerIcons.load(OutlineIcon.ROCKET, size=size, color=color_hex)
        elif entity_type == "code_zone":
            pil_img = TablerIcons.load(OutlineIcon.BARCODE, size=size, color=color_hex)
        elif entity_type == "station":
            pil_img = TablerIcons.load(OutlineIcon.SQUARE, size=size, color=color_hex)
        elif entity_type == "stage_start":
            pil_img = TablerIcons.load(OutlineIcon.PLAYER_PLAY, size=size, color=color_hex)
        elif entity_type == "stage_exit":
            pil_img = TablerIcons.load(OutlineIcon.PLAYER_STOP, size=size, color=color_hex)
        else:
            pil_img = TablerIcons.load(OutlineIcon.QUESTION_MARK, size=size, color=color_hex)
            
        surf = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode).convert_alpha()
        
    _IMAGE_CACHE[cache_key] = surf
    return surf


class DeferredMaintenanceEnv(gym.Env):
    """Deferred Maintenance task (protocol v2 semantics).

    Episode structure. A seeded schedule is drawn at reset: a readiness stage r, and one
    distinct code type for each of the first r stages ("collection stages"). Readiness
    switches on when the rocket enters stage r. Every stage shows one station per station
    type: the issued code types, plus every other type when distractor_stations=True. The
    station layout never depends on which codes the rocket holds. Every issued code can
    therefore be serviced in any stage from r on, so a perfect run is always possible.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,                # 'human' for Pygame rendering, None for headless
        spatial_sensor_type='local_grid',# Type of micro-navigation sensor ('local_grid' or 'raycasts')
        gravity_scale=500.0,             # Strength of the constant downward physics gravity
        path_complexity=0.5,             # Scales the minimum procedural random-walk length (max(10, 5*pc))
        max_active_requests=3,           # Maximum number of codes the agent can hold at once
        duration_matching_mode=False,    # Not implemented in v2 (raises if True)
        distractor_stations=False,       # If True, stations for code types not issued this episode also appear
        readiness_delay_range=(1, 2),    # Inclusive [min, max] number of stages traversed before readiness=1.0
        tilt_speed=50000.0,              # Angular velocity scaling for rocket rotation
        enable_air_brakes=True,          # Enables holding DOWN to immediately zero out velocity
        required_hover_duration=2.0,     # Time (seconds) agent must hover over a station to service it
        min_stage_width=10,              # Minimum grid width for generated stages
        min_stage_height=10,             # Minimum grid height for generated stages
        max_stages=4,                    # Number of stages; exiting the last one ends the episode
        max_code_types=3,                # Number of code/station types
        codes_per_collection_stage=1,    # Code zones placed in each pre-readiness stage

        enforce_boundaries=True,         # If True, boundaries act as physical walls. If False, they are sensors.
        allow_duplicate_codes=False,     # If True, agent can collect multiple codes of the same type in its inventory
        max_steps=4000,          # Episode limit (truncation); >= 2x the oracle's p95 length (1,859). None disables it
        debug_info=False,                # If True, info["debug"] exposes hidden task state (tests/oracles only)

        # --- REWARDS & PENALTIES ---
        reward_service_station=100.0,
        reward_perfect_completion=1000.0,
        penalty_premature_service=-10.0,
        penalty_wrong_station=-10.0,
        penalty_off_path=-0.1,
        penalty_leftover_codes=-1000.0,
        penalty_touch_boundary=-5.0
    ):
        super().__init__()
        if spatial_sensor_type not in ("local_grid", "raycasts"):
            raise ValueError("spatial_sensor_type must be 'local_grid' or 'raycasts'")
        if duration_matching_mode:
            raise NotImplementedError("duration_matching_mode is not implemented in protocol v2")
        if max_stages < 2:
            raise ValueError("max_stages must be >= 2")
        lo, hi = int(readiness_delay_range[0]), int(readiness_delay_range[1])
        if not (1 <= lo <= hi <= max_stages - 1):
            raise ValueError(
                f"readiness_delay_range={readiness_delay_range} must satisfy 1 <= min <= max <= max_stages-1 "
                f"({max_stages - 1}); otherwise readiness could never switch on before the final exit")
        if hi * codes_per_collection_stage > max_code_types:
            raise ValueError("readiness max * codes_per_collection_stage exceeds max_code_types (codes must be distinct)")
        if hi * codes_per_collection_stage > max_active_requests:
            raise ValueError("readiness max * codes_per_collection_stage exceeds max_active_requests (codes could not all be held)")
        if codes_per_collection_stage < 1:
            raise ValueError("codes_per_collection_stage must be >= 1")

        self.render_mode = render_mode
        self.spatial_sensor_type = spatial_sensor_type
        self.gravity_scale = gravity_scale
        self.max_active_requests = max_active_requests
        self.duration_matching_mode = duration_matching_mode
        self.distractor_stations = distractor_stations
        self.readiness_delay_range = (lo, hi)
        self.tilt_speed = tilt_speed
        self.enable_air_brakes = enable_air_brakes
        self.path_complexity = path_complexity
        self.required_hover_duration = required_hover_duration
        self.min_stage_width = min_stage_width
        self.min_stage_height = min_stage_height
        self.max_stages = max_stages
        self.max_code_types = max_code_types
        self.codes_per_collection_stage = codes_per_collection_stage
        self.enforce_boundaries = enforce_boundaries
        self.allow_duplicate_codes = allow_duplicate_codes
        self.max_steps = max_steps
        self.debug_info = debug_info

        self.reward_service_station = reward_service_station
        self.reward_perfect_completion = reward_perfect_completion
        self.penalty_premature_service = penalty_premature_service
        self.penalty_wrong_station = penalty_wrong_station
        self.penalty_off_path = penalty_off_path
        self.penalty_leftover_codes = penalty_leftover_codes
        self.penalty_touch_boundary = penalty_touch_boundary

        self.global_code_types = list(range(1, max_code_types + 1))
        self._prev_brake_engaged = False
        self.dt = 1 / 60.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        micro_nav_size = 9 if self.spatial_sensor_type == 'local_grid' else 8
        self.observation_space = spaces.Dict({
            'pos_x': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'pos_y': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'vel_x': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'vel_y': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'angle': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'ang_vel': spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            'micro_nav': spaces.Box(0.0, float(max_code_types + 10), shape=(micro_nav_size,), dtype=np.float32),
            'macro_nav': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            'floor_code': spaces.Box(0.0, float(max_code_types), shape=(), dtype=np.float32),
            'readiness': spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            'station_radar': spaces.Box(-np.inf, np.inf, shape=(2 * max_code_types,), dtype=np.float32),
        })

        self.screen_width = 800
        self.screen_height = 600
        self.physics_width = 500
        self.cell_size = 50
        self.screen = None
        self.clock = None

        self.space = None
        self.held_codes = []
        self.uncollected_codes = 0
        self.pending_code_removals = []
        self.reached_exit = False

        self.score = 0.0
        self.stages_traversed = 0
        self.readiness = 0.0
        self.readiness_target = lo
        self.issued_codes = []
        self.stage_code_types = {}
        self.station_types = []
        self.serviced_codes = []
        self.step_count = 0

        self.hover_timer = 0.0
        self.hovering_station_id = None
        self.touching_floor_code = 0.0
        self.rocket = None
        self.entities = []
        self.shape_to_entity = {}
        self.cell_types = {}
        self.path_cells = []
        self.path_index = {}
        self.stage_layout = None

    # ------------------------------------------------------------------ generation
    def _random_walk(self, length):
        rng = self.np_random
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        curr = (0, 0)
        coords = [curr]
        visited = {curr}
        for _ in range(length):
            moved = False
            for i in rng.permutation(4):
                nxt = (curr[0] + moves[i][0], curr[1] + moves[i][1])
                if nxt not in visited:
                    curr = nxt
                    visited.add(curr)
                    coords.append(curr)
                    moved = True
                    break
            if not moved:
                break
        return coords

    @staticmethod
    def _cheb(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _generate_procedural_stage(self, stage_index):
        """Return (stage_array, path_coords) for one stage of the episode schedule.

        Interactables (stations and codes) sit on interior path tiles, at least two cells
        (Chebyshev) from each other and from the exit, so hovering on one never touches
        another or the exit. The walk is redrawn (longer every 20 attempts) until they fit.
        """
        rng = self.np_random
        codes = list(self.stage_code_types.get(stage_index, []))
        items = [("station", t) for t in self.station_types] + [("code", t) for t in codes]
        base_length = max(10, int(5 * self.path_complexity))
        for attempt in range(400):
            coords = self._random_walk(base_length + 2 * (attempt // 20))
            if len(coords) < len(items) + 2:
                continue
            exit_c = coords[-1]
            interior = coords[1:-1]
            placed = []
            for idx in rng.permutation(len(interior)):
                if len(placed) == len(items):
                    break
                tile = interior[idx]
                if self._cheb(tile, exit_c) < 2:
                    continue
                if any(self._cheb(tile, p) < 2 for p, _ in placed):
                    continue
                placed.append((tile, items[len(placed)]))
            if len(placed) == len(items):
                break
        else:
            raise RuntimeError("could not generate a stage layout")

        grid_dict = {c: 1 for c in coords}
        grid_dict[coords[0]] = 4
        grid_dict[exit_c] = 5
        for tile, (kind, t) in placed:
            grid_dict[tile] = f"{kind}_{t}"

        min_x = min(x for x, y in grid_dict)
        max_x = max(x for x, y in grid_dict)
        min_y = min(y for x, y in grid_dict)
        max_y = max(y for x, y in grid_dict)
        grid_w = max_x - min_x + 1
        grid_h = max_y - min_y + 1
        cols = max(grid_w + 2, self.min_stage_width)
        rows = max(grid_h + 2, self.min_stage_height)
        off_x = (cols - grid_w) // 2 - min_x
        off_y = (rows - grid_h) // 2 - min_y
        stage_array = [[0 for _ in range(cols)] for _ in range(rows)]
        for (x, y), val in grid_dict.items():
            stage_array[y + off_y][x + off_x] = val
        # +1 for the boundary ring added in generate_stage
        path = [(x + off_x + 1, y + off_y + 1) for x, y in coords]
        return stage_array, path

    def generate_stage(self):
        for ent in self.entities:
            if ent.body in self.space.bodies:
                self.space.remove(ent.body, ent.shape)
        self.entities.clear()
        self.shape_to_entity = {}
        self.cell_types = {}

        stage_layout, path = self._generate_procedural_stage(self.stages_traversed)
        cols = len(stage_layout[0])
        rows = len(stage_layout)
        bordered_layout = [[9] * (cols + 2)]
        for row in stage_layout:
            bordered_layout.append([9] + list(row) + [9])
        bordered_layout.append([9] * (cols + 2))
        self.stage_layout = bordered_layout
        self.path_cells = path
        self.path_index = {c: i for i, c in enumerate(path)}
        self._path_centers = np.array(
            [((c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size) for c, r in path], dtype=np.float64)

        spawn_pos = (self.cell_size * 1.5, self.screen_height // 2)
        for row_idx, row in enumerate(bordered_layout):
            for col_idx, val in enumerate(row):
                x = col_idx * self.cell_size
                y = row_idx * self.cell_size
                rect = (x, y, self.cell_size, self.cell_size)
                ent = None
                ctype = CELL_VOID
                if val == 1:
                    ent = PathBoundary(self.space, rect); ctype = CELL_PATH
                elif isinstance(val, str) and val.startswith("station_"):
                    ent = MaintenanceStation(self.space, rect, color_code=int(val.split("_")[1])); ctype = CELL_STATION
                elif isinstance(val, str) and val.startswith("code_"):
                    self.uncollected_codes += 1
                    ent = CodeZone(self.space, rect, color_code=int(val.split("_")[1])); ctype = CELL_CODE
                elif val == 4:
                    ent = StageEntry(self.space, rect); ctype = CELL_ENTRY
                    spawn_pos = (x + self.cell_size / 2, y + self.cell_size / 2)
                elif val == 5:
                    ent = StageExit(self.space, rect); ctype = CELL_EXIT
                elif val == 9:
                    ent = StageBoundary(self.space, rect, is_solid=self.enforce_boundaries); ctype = CELL_BOUNDARY
                if ent is not None:
                    self.entities.append(ent)
                    self.shape_to_entity[ent.shape] = ent
                self.cell_types[(col_idx, row_idx)] = ctype

        if self.rocket is None:
            self.rocket = Rocket(self.space, spawn_pos)
            self.rocket.shape.filter = pymunk.ShapeFilter(categories=ROCKET_CATEGORY)
        else:
            self.rocket.body.position = spawn_pos
            self.rocket.body.velocity = (0, 0)
            self.rocket.body.angular_velocity = 0
            self.rocket.body.angle = 0
        self.hovering_station_id = None
        self.hover_timer = 0.0

    # ------------------------------------------------------------------ gym API
    def _debug(self):
        return {
            "held_codes": list(self.held_codes),
            "serviced_codes": list(self.serviced_codes),
            "issued_codes": list(self.issued_codes),
            "readiness_target": int(self.readiness_target),
            "stage": int(self.stages_traversed),
            "uncollected_codes": int(self.uncollected_codes),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        # A fresh physics space per episode, so no solver/contact state leaks across episodes.
        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity_scale)

        lo, hi = self.readiness_delay_range
        self.readiness_target = int(rng.integers(lo, hi + 1))
        n_issued = self.readiness_target * self.codes_per_collection_stage
        self.issued_codes = [int(t) + 1 for t in rng.choice(self.max_code_types, size=n_issued, replace=False)]
        cps = self.codes_per_collection_stage
        self.stage_code_types = {s: self.issued_codes[s * cps:(s + 1) * cps] for s in range(self.readiness_target)}
        self.station_types = list(self.global_code_types) if self.distractor_stations else sorted(self.issued_codes)

        self.held_codes = []
        self.serviced_codes = []
        self.uncollected_codes = 0
        self.pending_code_removals = []
        self.reached_exit = False
        self.score = 0.0
        self.stages_traversed = 0
        self.readiness = 0.0
        self.step_count = 0
        self.hover_timer = 0.0
        self.hovering_station_id = None
        self.touching_floor_code = 0.0
        self._prev_brake_engaged = False
        self.rocket = None
        self.entities = []
        self.generate_stage()

        if self.render_mode == "human":
            self.render()

        info = {"events": []}
        if self.debug_info:
            info["debug"] = self._debug()
        return self._get_obs(), info

    def _cell_of(self, pos):
        return (int(math.floor(pos[0] / self.cell_size)), int(math.floor(pos[1] / self.cell_size)))

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        thrust = float(np.clip(action[0], -1.0, 1.0))
        torque_input = float(np.clip(action[1], -1.0, 1.0))

        brake_engaged = (thrust < -0.5)

        if self.enable_air_brakes and brake_engaged and not self._prev_brake_engaged:
            self.rocket.body.velocity = (0.0, self.rocket.body.velocity.y)
            self.rocket.body.angular_velocity = 0.0

        self._prev_brake_engaged = brake_engaged

        if thrust > 0:
            self.rocket.apply_thrust(thrust * 2000.0)

        max_angle = math.pi / 2.0
        current_angle = self.rocket.body.angle

        if abs(torque_input) > 0.05:
            self.rocket.apply_rotation(torque_input * self.tilt_speed)
            self.rocket.body.angular_velocity *= 0.85
        else:
            restoring_torque = -current_angle * (self.tilt_speed * 0.8)
            self.rocket.apply_rotation(restoring_torque)
            self.rocket.body.angular_velocity *= 0.85

            if abs(current_angle) < 0.05 and abs(self.rocket.body.angular_velocity) < 0.1:
                self.rocket.body.angle = 0.0
                self.rocket.body.angular_velocity = 0.0

        if self.rocket.body.angle > max_angle:
            self.rocket.body.angle = max_angle
            self.rocket.body.angular_velocity = min(self.rocket.body.angular_velocity, 0.0)
        elif self.rocket.body.angle < -max_angle:
            self.rocket.body.angle = -max_angle
            self.rocket.body.angular_velocity = max(self.rocket.body.angular_velocity, 0.0)

        self.space.step(self.dt)
        self.step_count += 1

        reward = 0.0
        terminated = False
        events = []
        cause = None

        touching_station_id = None
        touching_code_id = None
        touching_boundary = False
        reached_exit = False

        for q in self.space.shape_query(self.rocket.shape):
            ent = self.shape_to_entity.get(q.shape)
            if ent is None:
                continue
            if isinstance(ent, MaintenanceStation):
                touching_station_id = ent.color_code
            elif isinstance(ent, CodeZone):
                touching_code_id = ent.color_code
                if ent in self.pending_code_removals:
                    continue
                if len(self.held_codes) < self.max_active_requests and (
                        self.allow_duplicate_codes or ent.color_code not in self.held_codes):
                    self.held_codes.append(ent.color_code)
                    self.pending_code_removals.append(ent)
                    events.append(("code_collected", int(ent.color_code)))
            elif isinstance(ent, StageExit):
                reached_exit = True
            elif isinstance(ent, StageBoundary):
                touching_boundary = True

        self.touching_floor_code = float(touching_station_id or touching_code_id or 0.0)

        if touching_boundary:
            reward += self.penalty_touch_boundary
            events.append(("boundary_contact", None))

        if touching_station_id is not None:
            if self.hovering_station_id != touching_station_id:
                self.hovering_station_id = touching_station_id
                self.hover_timer = 0.0
            self.hover_timer += self.dt
            # small tolerance so 2.0 s means exactly 120 steps at 60 Hz
            if self.hover_timer >= self.required_hover_duration - 1e-9:
                k = int(self.hovering_station_id)
                events.append(("service_attempt", k))
                if self.readiness == 1.0 and k in self.held_codes:
                    reward += self.reward_service_station
                    self.held_codes.remove(k)
                    self.serviced_codes.append(k)
                elif self.readiness < 1.0:
                    reward += self.penalty_premature_service
                else:
                    reward += self.penalty_wrong_station
                self.hover_timer = 0.0
        else:
            self.hovering_station_id = None
            self.hover_timer = 0.0

        if self.cell_types.get(self._cell_of(self.rocket.body.position), CELL_VOID) not in SAFE_CELLS:
            reward += self.penalty_off_path
            events.append(("off_path", None))

        # Collected code zones turn into plain path tiles.
        for ent in self.pending_code_removals:
            if ent in self.entities:
                if ent.body in self.space.bodies:
                    self.space.remove(ent.body, ent.shape)
                self.entities.remove(ent)
                self.shape_to_entity.pop(ent.shape, None)
                tile = PathBoundary(self.space, ent.rect)
                self.entities.append(tile)
                self.shape_to_entity[tile.shape] = tile
                self.cell_types[(ent.rect[0] // self.cell_size, ent.rect[1] // self.cell_size)] = CELL_PATH
                self.uncollected_codes -= 1
        self.pending_code_removals.clear()

        if reached_exit:
            if self.uncollected_codes > 0:
                events.append(("stage_exit_incomplete", int(self.stages_traversed)))
                reward += self.penalty_leftover_codes
                terminated = True
                cause = "uncollected_codes_at_exit"
            else:
                events.append(("stage_exit", int(self.stages_traversed)))
                self.stages_traversed += 1
                if self.stages_traversed >= self.max_stages:
                    events.append(("final_exit", None))
                    if len(self.held_codes) > 0:
                        reward += self.penalty_leftover_codes
                        cause = "held_codes_at_final_exit"
                    else:
                        reward += self.reward_perfect_completion
                        cause = "success"
                    terminated = True
                else:
                    if self.stages_traversed >= self.readiness_target and self.readiness < 1.0:
                        self.readiness = 1.0
                        events.append(("readiness_on", None))
                    self.generate_stage()

        # score is the running sum of step rewards (each reward counted once)
        self.score += reward

        truncated = False
        if not terminated and self.max_steps is not None and self.step_count >= self.max_steps:
            truncated = True
            cause = "timeout"

        if self.render_mode == "human":
            self.render()

        info = {"events": events}
        if terminated or truncated:
            info["termination_cause"] = cause
            info["is_success"] = cause == "success"
        if self.debug_info:
            info["debug"] = self._debug()
        return self._get_obs(), float(reward), terminated, truncated, info

    def waypoint(self):
        """Offset from the rocket to the centre of the next path tile towards the exit."""
        pos = np.array([self.rocket.body.position.x, self.rocket.body.position.y])
        idx = self.path_index.get(self._cell_of(pos))
        if idx is None:
            idx = int(np.argmin(((self._path_centers - pos) ** 2).sum(axis=1)))
        target = self._path_centers[min(idx + 1, len(self._path_centers) - 1)]
        return target - pos

    def _get_obs(self):
        obs = {}
        pos = self.rocket.body.position
        obs['pos_x'] = float(pos.x)
        obs['pos_y'] = float(pos.y)
        obs['vel_x'] = float(self.rocket.body.velocity.x)
        obs['vel_y'] = float(self.rocket.body.velocity.y)
        obs['angle'] = float(self.rocket.body.angle)
        obs['ang_vel'] = float(self.rocket.body.angular_velocity)

        if self.spatial_sensor_type == 'local_grid':
            # Cell type of the 3x3 neighbourhood (same codes as the collision types).
            grid = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    cell = self._cell_of((pos.x + dx * self.cell_size, pos.y + dy * self.cell_size))
                    grid.append(float(self.cell_types.get(cell, CELL_VOID)))
            obs['micro_nav'] = grid
        else:
            ray_distances = []
            for i in range(8):
                angle = i * (2 * math.pi / 8)
                direction = pymunk.Vec2d(1, 0).rotated(angle)
                end = pos + direction * 150.0 # 150px sensor range
                res = self.space.segment_query_first(pos, end, 1, SENSOR_FILTER)
                ray_distances.append(float(res.alpha) if res else 1.0)
            obs['micro_nav'] = ray_distances

        obs['macro_nav'] = [float(v) for v in self.waypoint()]
        obs['floor_code'] = float(self.touching_floor_code)
        obs['readiness'] = float(self.readiness)

        radar = []
        for v in self.global_code_types:
            best = None
            for e in self.entities:
                if isinstance(e, MaintenanceStation) and e.color_code == v:
                    d = (e.body.position.x - pos.x, e.body.position.y - pos.y)
                    if best is None or d[0] ** 2 + d[1] ** 2 < best[0] ** 2 + best[1] ** 2:
                        best = d
            radar.extend(best if best is not None else [-99.0, -99.0])
        obs['station_radar'] = [float(v) for v in radar]

        return obs

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))
        
        cam_x = self.rocket.body.position.x - self.physics_width / 2
        cam_y = self.rocket.body.position.y - self.screen_height / 2
        
        pygame.draw.rect(self.screen, (20, 20, 20), (0, 0, self.physics_width, self.screen_height))
        
        font = pygame.font.SysFont(None, 24)
        
        for ent in self.entities:
            draw_rect = (ent.rect[0] - cam_x, ent.rect[1] - cam_y, ent.rect[2], ent.rect[3])
            
            if isinstance(ent, PathBoundary):
                pygame.draw.rect(self.screen, (255, 255, 255), draw_rect)
            elif isinstance(ent, StageBoundary):
                pygame.draw.rect(self.screen, (0, 0, 0), draw_rect)
                pygame.draw.rect(self.screen, (255, 255, 255), draw_rect, 4) # Hollow thick white border
            elif isinstance(ent, CodeZone):
                hex_col = get_color_hex(ent.color_code)
                pygame.draw.rect(self.screen, (40, 40, 40), draw_rect)
                icon = get_icon("code_zone", hex_col, 32)
                self.screen.blit(icon, (draw_rect[0] + 9, draw_rect[1] + 9))
                
                t_surf = font.render(str(ent.color_code), True, (255, 255, 255))
                self.screen.blit(t_surf, (draw_rect[0] + 4, draw_rect[1] + 4))
                
            elif isinstance(ent, MaintenanceStation):
                hex_col = get_color_hex(ent.color_code)
                pygame.draw.rect(self.screen, (40, 40, 40), draw_rect)
                icon = get_icon("station", hex_col, 32)
                self.screen.blit(icon, (draw_rect[0] + 9, draw_rect[1] + 9))
                
                t_surf = font.render(str(ent.color_code), True, (255, 255, 255))
                self.screen.blit(t_surf, (draw_rect[0] + 4, draw_rect[1] + 4))
                
            elif isinstance(ent, StageEntry):
                pygame.draw.rect(self.screen, (150, 150, 150), draw_rect)
                icon = get_icon("stage_start", "#FFFFFF", 32)
                self.screen.blit(icon, (draw_rect[0] + 9, draw_rect[1] + 9))
            elif isinstance(ent, StageExit):
                pygame.draw.rect(self.screen, (100, 255, 100), draw_rect)
                icon = get_icon("stage_exit", "#000000", 32)
                self.screen.blit(icon, (draw_rect[0] + 9, draw_rect[1] + 9))

        pos = self.rocket.body.position
        screen_pos_x = pos.x - cam_x
        screen_pos_y = pos.y - cam_y
        
        angle = self.rocket.body.angle
        rocket_icon = get_icon("rocket", "#FF0000", 40)
        rotated_rocket = pygame.transform.rotate(rocket_icon, -math.degrees(angle))
        rect = rotated_rocket.get_rect(center=(int(screen_pos_x), int(screen_pos_y)))
        self.screen.blit(rotated_rocket, rect.topleft)

        panel_x = self.physics_width
        pygame.draw.rect(self.screen, (50, 50, 50), (panel_x, 0, self.screen_width - panel_x, self.screen_height))
        
        held_names = [f"Type {c}" for c in self.held_codes]
        
        stats = [
            "--- LEGEND / STATS ---",
            f"Health / Score: {self.score:.1f}",
            f"Action Readiness: {'1.0 (READY)' if self.readiness == 1.0 else '0.0 (WAITING)'}",
            f"Held Codes: {held_names}",
            f"Hover Timer: {self.hover_timer:.1f}s",
            f"Stages: {self.stages_traversed}/{self.max_stages}",
            f"Uncollected Map Codes: {self.uncollected_codes}"
        ]
        
        for i, text in enumerate(stats):
            img = font.render(text, True, (255, 255, 255))
            self.screen.blit(img, (panel_x + 10, 20 + i * 30))
            
        legend_items = [
            ("rocket", "Rocket Agent", "#FF0000"),
            ("code_zone", "Maintenance Code", "#FFFFFF"),
            ("station", "Maintenance Station", "#FFFFFF"),
            ("stage_start", "Stage Start", "#FFFFFF"),
            ("stage_exit", "Stage Exit", "#000000")
        ]
        
        y_offset = 250
        for e_type, desc, color in legend_items:
            icon = get_icon(e_type, color, 32)
            self.screen.blit(icon, (panel_x + 10, y_offset))
            img = font.render(desc, True, (200, 200, 200))
            self.screen.blit(img, (panel_x + 50, y_offset + 8))
            y_offset += 40

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
