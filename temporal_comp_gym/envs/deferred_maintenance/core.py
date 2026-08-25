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

# ---------------------------------------------------------
# CUSTOM_STAGES
# ---------------------------------------------------------
# To populate CUSTOM_STAGES, use a 2D array representing the grid:
# 0 = Black Square (Dangerous Space)
# 1 = White Square (Safe Path)
# 4 = Stage Entry/Spawn Point
# 5 = Stage Exit (Teleport to next stage)
# "code_X"    = Code Zone of type X (e.g., "code_1", "code_2")
# "station_X" = Maintenance Station of type X (e.g., "station_1")
CUSTOM_STAGES = [
    # Stage 1: Simple 3x10 path
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 1, "code_1", 1, "station_1", 1, 1, 1, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Stage 2: Vertical winding 7x5
    [
        [0, 0, 0, 0, 0],
        [0, 4, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, "code_2", 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, "station_2", 1, 0],
        [0, 0, 0, 5, 0]
    ]
]

CUSTOM_STAGES = []

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
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,                # 'human' for Pygame rendering, None for headless
        spatial_sensor_type='local_grid',# Type of micro-navigation sensor ('local_grid' or 'raycasts')
        gravity_scale=500.0,             # Strength of the constant downward physics gravity
        path_complexity=0.5,               # Multiplier for the procedural random-walk length
        max_active_requests=3,           # Maximum number of codes the agent can hold at once
        duration_matching_mode=False,    # Legacy toggle for alternate durations
        distractor_stations=False,       # Spawns stations that don't match any code
        readiness_delay_range=(1, 3),    # Range of stages agent must traverse before readiness=1.0
        tilt_speed=50000.0,              # Angular velocity scaling for rocket rotation
        enable_air_brakes=True,          # Enables holding DOWN to immediately zero out velocity
        required_hover_duration=2.0,     # Time (seconds) agent must hover over a station to service it
        min_stage_width=10,              # Minimum grid width for auto-generated stages (ignored by CUSTOM_STAGES)
        min_stage_height=10,             # Minimum grid height for auto-generated stages (ignored by CUSTOM_STAGES)
        max_stages=4,                    # Maximum number of stages before final exit. Overrides CUSTOM_STAGES length.
        max_code_types=3,                # Maximum number of unique code/station types allowed in the entire game
        
        enforce_boundaries=True,         # If True, boundaries act as physical walls. If False, they are sensors.
        allow_duplicate_codes=False,     # If True, agent can collect multiple codes of the same type in its inventory
        
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
        self.render_mode = render_mode
        self.spatial_sensor_type = spatial_sensor_type
        self.gravity_scale = gravity_scale
        self.max_active_requests = max_active_requests
        self.duration_matching_mode = duration_matching_mode
        self.distractor_stations = distractor_stations
        self.readiness_delay_range = readiness_delay_range
        self.tilt_speed = tilt_speed
        self.enable_air_brakes = enable_air_brakes
        self.path_complexity = path_complexity
        self.required_hover_duration = required_hover_duration
        self.min_stage_width = min_stage_width
        self.min_stage_height = min_stage_height
        self.max_stages = max_stages
        self.max_code_types = max_code_types
        self.enforce_boundaries = enforce_boundaries
        self.allow_duplicate_codes = allow_duplicate_codes
        
        self.reward_service_station = reward_service_station
        self.reward_perfect_completion = reward_perfect_completion
        self.penalty_premature_service = penalty_premature_service
        self.penalty_wrong_station = penalty_wrong_station
        self.penalty_off_path = penalty_off_path
        self.penalty_leftover_codes = penalty_leftover_codes
        self.penalty_touch_boundary = penalty_touch_boundary
        
        self.global_code_types = list(range(1, max_code_types + 1))
        self._prev_brake_engaged = False
        
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
        
        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity_scale)
        
        self.held_codes = []
        self.uncollected_codes = 0
        self.pending_code_removals = []
        self.reached_exit = False
        
        self.score = 0.0
        self.stages_traversed = 0
        self.readiness = 0.0
        self.readiness_target = np.random.randint(*self.readiness_delay_range) if self.readiness_delay_range[1] > self.readiness_delay_range[0] else self.readiness_delay_range[0]
        
        self.hover_timer = 0.0
        self.hovering_station_id = None
        self.touching_floor_code = 0.0
        self.rocket = None
        self.entities = []
        
    def _generate_procedural_stage(self):
        length = max(10, 5 * self.path_complexity)
        grid_dict = {}
        curr = (0, 0)
        grid_dict[curr] = 4 # Entry
        
        path_coords = [curr]
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for _ in range(length):
            np.random.shuffle(moves)
            for m in moves:
                nxt = (curr[0] + m[0], curr[1] + m[1])
                if nxt not in grid_dict:
                    curr = nxt
                    grid_dict[curr] = 1 
                    path_coords.append(curr)
                    break
                    
        safe_tiles = [k for k, v in grid_dict.items() if v == 1]
        np.random.shuffle(safe_tiles)
        
        num_codes = max(1, self.path_complexity)
        chosen_types = [np.random.choice(self.global_code_types) for _ in range(num_codes)]
        
        for c_type in chosen_types:
            if safe_tiles: grid_dict[safe_tiles.pop()] = f"code_{c_type}"
            
        stations_per_code = max(1, 4 - self.stages_traversed)
        for c_type in chosen_types:
            for _ in range(stations_per_code):
                if safe_tiles: grid_dict[safe_tiles.pop()] = f"station_{c_type}"
            
        grid_dict[path_coords[-1]] = 5
        
        min_x = min(x for x, y in grid_dict.keys())
        max_x = max(x for x, y in grid_dict.keys())
        min_y = min(y for x, y in grid_dict.keys())
        max_y = max(y for x, y in grid_dict.keys())
        
        grid_w = max_x - min_x + 1
        grid_h = max_y - min_y + 1
        
        cols = max(grid_w + 2, self.min_stage_width)
        rows = max(grid_h + 2, self.min_stage_height)
        
        off_x = (cols - grid_w) // 2 - min_x
        off_y = (rows - grid_h) // 2 - min_y
        
        stage_array = [[0 for _ in range(cols)] for _ in range(rows)]
        
        for (x, y), val in grid_dict.items():
            stage_array[y + off_y][x + off_x] = val
            
        return stage_array

    def generate_stage(self):
        for ent in self.entities:
            if ent.body in self.space.bodies:
                self.space.remove(ent.body, ent.shape)
        self.entities.clear()
        
        # Determine layout
        if self.stages_traversed < len(CUSTOM_STAGES):
            stage_layout = CUSTOM_STAGES[self.stages_traversed]
        else:
            stage_layout = self._generate_procedural_stage()
            
        # Add visual boundary wrapping (9s) tightly around the layout
        cols = len(stage_layout[0])
        rows = len(stage_layout)
        bordered_layout = [[9] * (cols + 2)]
        for row in stage_layout:
            bordered_layout.append([9] + list(row) + [9])
        bordered_layout.append([9] * (cols + 2))
        stage_layout = bordered_layout
            
        spawn_pos = (self.cell_size * 1.5, self.screen_height // 2)
        
        for row_idx, row in enumerate(stage_layout):
            for col_idx, val in enumerate(row):
                x = col_idx * self.cell_size
                y = row_idx * self.cell_size
                rect = (x, y, self.cell_size, self.cell_size)
                
                if val == 1:
                    self.entities.append(PathBoundary(self.space, rect))
                elif isinstance(val, str) and val.startswith("station_"):
                    type_id = int(val.split("_")[1])
                    self.entities.append(MaintenanceStation(self.space, rect, color_code=type_id))
                elif isinstance(val, str) and val.startswith("code_"):
                    type_id = int(val.split("_")[1])
                    self.uncollected_codes += 1
                    self.entities.append(CodeZone(self.space, rect, color_code=type_id))
                elif val == 4:
                    self.entities.append(StageEntry(self.space, rect))
                    spawn_pos = (x + self.cell_size/2, y + self.cell_size/2)
                elif val == 5:
                    self.entities.append(StageExit(self.space, rect))
                elif val == 9:
                    self.entities.append(StageBoundary(self.space, rect, is_solid=self.enforce_boundaries))
        
        if self.rocket is None:
            self.rocket = Rocket(self.space, spawn_pos)
        else:
            self.rocket.body.position = spawn_pos
            self.rocket.body.velocity = (0, 0)
            self.rocket.body.angular_velocity = 0
            self.rocket.body.angle = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.held_codes = []
        self.uncollected_codes = 0
        self.pending_code_removals = []
        self.reached_exit = False
        
        self.score = 0.0
        self.stages_traversed = 0
        self.readiness = 0.0
        self.readiness_target = np.random.randint(*self.readiness_delay_range) if self.readiness_delay_range[1] > self.readiness_delay_range[0] else self.readiness_delay_range[0]
        self.hover_timer = 0.0
        self.hovering_station_id = None
        self.touching_floor_code = 0.0
        self._prev_brake_engaged = False
        
        if self.rocket is not None:
            if self.rocket.body in self.space.bodies:
                self.space.remove(self.rocket.body, self.rocket.shape)
            self.rocket = None
            
        self.generate_stage()
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}
        
    def step(self, action):
        thrust = np.clip(action[0], -1.0, 1.0)
        torque_input = np.clip(action[1], -1.0, 1.0)
        
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
            
        self.space.step(1/60.0)
        
        reward = 0.0
        terminated = False
        
        # --- CONTINUOUS SHAPE QUERY FOR ROBUST TRIGGER OVERLAPS ---
        touching_station_id = None
        touching_code_id = None
        touching_boundary = False
        
        for info in self.space.shape_query(self.rocket.shape):
            shape = info.shape
            ent = next((e for e in self.entities if hasattr(e, 'shape') and e.shape == shape), None)
            if ent:
                if isinstance(ent, MaintenanceStation):
                    touching_station_id = ent.color_code
                elif isinstance(ent, CodeZone):
                    touching_code_id = ent.color_code
                    if len(self.held_codes) < self.max_active_requests:
                        can_collect = True
                        if not self.allow_duplicate_codes and ent.color_code in self.held_codes:
                            can_collect = False
                            
                        if can_collect:
                            self.held_codes.append(ent.color_code)
                            if ent not in self.pending_code_removals:
                                self.pending_code_removals.append(ent)
                elif isinstance(ent, StageExit):
                    self.reached_exit = True
                elif isinstance(ent, StageBoundary):
                    touching_boundary = True
                    
        self.touching_floor_code = touching_station_id or touching_code_id or 0.0

        if touching_boundary:
            reward += self.penalty_touch_boundary
            self.score += self.penalty_touch_boundary

        if touching_station_id is not None:
            if self.hovering_station_id != touching_station_id:
                self.hovering_station_id = touching_station_id
                self.hover_timer = 0.0
                
            self.hover_timer += 1/60.0
            
            if self.hover_timer >= self.required_hover_duration:
                if self.readiness == 1.0 and self.hovering_station_id in self.held_codes:
                    reward += self.reward_service_station
                    self.score += reward
                    self.held_codes.remove(self.hovering_station_id)
                else:
                    if self.readiness < 1.0:
                        reward += self.penalty_premature_service
                        self.score += self.penalty_premature_service
                    elif self.hovering_station_id not in self.held_codes:
                        reward += self.penalty_wrong_station
                        self.score += self.penalty_wrong_station
                self.hover_timer = 0.0
        else:
            self.hovering_station_id = None
            self.hover_timer = 0.0
                
        info = self.space.point_query_nearest(self.rocket.body.position, 0, pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS()))
        safe_types = {2, 3, 4, 5, 6, 7} # include 7 (boundary) so we don't double penalize with off-path
        if info is None or info.shape.collision_type not in safe_types:
            reward += self.penalty_off_path
            self.score += self.penalty_off_path

        # Process Code Collections (turn them physically into white squares)
        for ent in self.pending_code_removals:
            if ent in self.entities:
                if ent.body in self.space.bodies:
                    self.space.remove(ent.body, ent.shape)
                self.entities.remove(ent)
                self.entities.append(PathBoundary(self.space, ent.rect))
                self.uncollected_codes -= 1
        self.pending_code_removals.clear()

        # Process Stage Exit
        if self.reached_exit:
            self.reached_exit = False
            
            if self.uncollected_codes > 0:
                reward += self.penalty_leftover_codes
                self.score += reward
                terminated = True
            else:
                self.stages_traversed += 1
                if self.stages_traversed >= self.max_stages:
                    if len(self.held_codes) > 0:
                        reward += self.penalty_leftover_codes
                    else:
                        reward += self.reward_perfect_completion
                    self.score += reward
                    terminated = True
                else:
                    if self.stages_traversed >= self.readiness_target:
                        self.readiness = 1.0 
                    self.generate_stage()

        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, False, {}
        
    def _get_obs(self):
        obs = {}
        obs['pos_x'] = float(self.rocket.body.position.x)
        obs['pos_y'] = float(self.rocket.body.position.y)
        obs['vel_x'] = float(self.rocket.body.velocity.x)
        obs['vel_y'] = float(self.rocket.body.velocity.y)
        obs['angle'] = float(self.rocket.body.angle)
        obs['ang_vel'] = float(self.rocket.body.angular_velocity)
        
        # Raycasts for micro_nav (agent perceives objects exactly the same way around it)
        ray_filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS())
        
        if self.spatial_sensor_type == 'local_grid':
            grid = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    pt = self.rocket.body.position + pymunk.Vec2d(dx * self.cell_size, dy * self.cell_size)
                    res = self.space.point_query_nearest(pt, 0, ray_filter)
                    if res and res.shape != self.rocket.shape:
                        grid.append(float(res.shape.collision_type))
                    else:
                        grid.append(0.0)
            obs['micro_nav'] = grid
        else:
            ray_distances = []
            for i in range(8):
                angle = i * (2 * math.pi / 8)
                direction = pymunk.Vec2d(1, 0).rotated(angle)
                start = self.rocket.body.position
                end = start + direction * 150.0 # 150px sensor range
                
                res = self.space.segment_query_first(start, end, 1, ray_filter)
                if res and res.shape != self.rocket.shape:
                    ray_distances.append(res.alpha) # 0.0 to 1.0 based on distance
                else:
                    ray_distances.append(1.0)
            obs['micro_nav'] = ray_distances
            
        obs['macro_nav'] = [0.0, 0.0]
        obs['floor_code'] = float(self.touching_floor_code)
        obs['readiness'] = self.readiness
        
        radar = []
        for v in self.global_code_types:
            station = next((e for e in self.entities if isinstance(e, MaintenanceStation) and e.color_code == v), None)
            if station:
                dx = station.body.position.x - self.rocket.body.position.x
                dy = station.body.position.y - self.rocket.body.position.y
                radar.extend([dx, dy])
            else:
                radar.extend([-99.0, -99.0])
        obs['station_radar'] = radar
        
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
