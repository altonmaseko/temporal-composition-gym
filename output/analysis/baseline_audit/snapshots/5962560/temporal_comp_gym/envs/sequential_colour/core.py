import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import io

try:
    import requests
    from PIL import Image
    HAS_EXTERNAL_DEPS = True
except ImportError:
    HAS_EXTERNAL_DEPS = False

try:
    from pytablericons import TablerIcons, OutlineIcon
except ImportError:
    TablerIcons = None
    OutlineIcon = None

from .objects import Agent, Room

# ==============================================================================
# EXTERNAL ICON URLS
# Insert external image URLs for these icons. If a URL string is provided for a 
# specific entity, it will fetch and render that image. If the string is empty 
# or missing, it will fallback to the pytablericons icon.
# ==============================================================================
ICON_URLS = {
    "agent": "",
    "target_wall": "",
    "penalty_wall": ""
}

# Pre-defined colours for vocabulary
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 165, 0), (255, 192, 203), (128, 128, 128), (255, 255, 255)
]

class SequentialColourEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self, 
        
        # Base Environment Settings
        render_mode=None,                               # Pygame rendering mode ('human', 'rgb_array', or None)
        
        # Difficulty & Logic Settings
        sequence_length_range=(3, 5),                   # Min/max colours flashed during memorisation
        retention_delay_steps=2,                        # Number of blank frames between memorisation and navigation
        allow_consecutive_duplicates=False,             # If True, the same colour can flash back-to-back
        colour_vocabulary_size=6,                       # Number of distinct colours the agent must learn
        randomise_wall_colors=True,                     # If True, randomises the 4 navigation wall colours every reset
        hard_death_mode=False,                          # If True, touching one incorrect wall ends the episode
        infinite_mode=False,                            # If True, immediately starts next sequence instead of terminating
        
        # Room Scaling
        agent_wall_distance_blocks=6,                   # Distance in blocks between the agent's spawn and the walls
        
        # Reward & Penalty Settings
        reward_density='sparse',                        # 'dense' grants reward per wall, 'sparse' requires full completion
        reward_correct_wall=1.0,                        # Reward granted when touching the correct wall (if density='dense')
        reward_sequence_completion=10.0,                # Terminal reward for finishing the sequence (if density='sparse')
        penalty_severity=1.0                            # Deduction applied to score and reward when touching an incorrect wall
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.sequence_length_range = sequence_length_range
        self.retention_delay_steps = retention_delay_steps
        self.allow_consecutive_duplicates = allow_consecutive_duplicates
        self.colour_vocabulary_size = min(colour_vocabulary_size, len(COLORS))
        self.randomise_wall_colors = randomise_wall_colors
        self.hard_death_mode = hard_death_mode
        self.infinite_mode = infinite_mode
        self.agent_wall_distance_blocks = agent_wall_distance_blocks
        
        self.reward_density = reward_density
        self.reward_correct_wall = reward_correct_wall
        self.reward_sequence_completion = reward_sequence_completion
        self.penalty_severity = penalty_severity
        
        self.vocab_colors = COLORS[:self.colour_vocabulary_size]
        
        # Action space: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 84x84x3 RGB image
        self.obs_size = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8
        )
        
        self.window_width = 1200
        self.window_height = 800
        self.game_size = 800
        
        self.window = None
        self.clock = None
        self.font = None
        
        # Calculate block sizes so that the room perfectly fits the requested distance
        # Total blocks = (Distance Left * 2) + 1 (Agent) + 2 (Walls)
        self.total_blocks = (self.agent_wall_distance_blocks * 2) + 3
        self.block_size = self.game_size / self.total_blocks
        
        self.agent = Agent(
            x=self.game_size/2, 
            y=self.game_size/2, 
            size=self.block_size, 
            speed=self.block_size/2  # Takes 2 steps to cross 1 block
        )
        self.room = Room(
            self.game_size, 
            self.game_size, 
            wall_thickness=self.block_size
        )
        
        self.icons = None
        # Icons will be loaded during the first render() call when video mode is available.
        
    def _load_icons(self):
        def get_surface(url, fallback_icon, fallback_color):
            if url and HAS_EXTERNAL_DEPS:
                try:
                    resp = requests.get(url, timeout=5)
                    img = Image.open(io.BytesIO(resp.content))
                    img = img.resize((64, 64))
                    img_data = io.BytesIO()
                    img.save(img_data, format='PNG')
                    img_data.seek(0)
                    surf = pygame.image.load(img_data).convert_alpha()
                    return surf
                except Exception as e:
                    print(f"Failed to load image from {url}: {e}")
            
            if TablerIcons is not None and fallback_icon is not None:
                try:
                    icon_pil = TablerIcons.load(fallback_icon, size=64, color=fallback_color)
                    img_data = io.BytesIO()
                    icon_pil.save(img_data, format='PNG')
                    img_data.seek(0)
                    return pygame.image.load(img_data).convert_alpha()
                except Exception as e:
                    print(f"Failed to load pytablericons {fallback_icon}: {e}")
            
            # Absolute fallback if no pytablericons or URL fails
            surf = pygame.Surface((64, 64), pygame.SRCALPHA)
            pygame.draw.circle(surf, fallback_color, (32, 32), 32)
            return surf

        self.icons["agent"] = get_surface(ICON_URLS.get("agent"), OutlineIcon.USER if TablerIcons else None, 'white')
        self.icons["target_wall"] = get_surface(ICON_URLS.get("target_wall"), OutlineIcon.TARGET if TablerIcons else None, 'green')
        self.icons["penalty_wall"] = get_surface(ICON_URLS.get("penalty_wall"), OutlineIcon.X if TablerIcons else None, 'red')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 10.0
        self.current_phase = "memorisation"
        self.step_count = 0
        
        seq_len = self.np_random.integers(self.sequence_length_range[0], self.sequence_length_range[1] + 1)
        self.sequence = []
        last_color_idx = -1
        
        for _ in range(seq_len):
            if self.allow_consecutive_duplicates:
                idx = self.np_random.integers(0, self.colour_vocabulary_size)
            else:
                choices = [i for i in range(self.colour_vocabulary_size) if i != last_color_idx]
                idx = self.np_random.choice(choices)
            self.sequence.append(self.vocab_colors[idx])
            last_color_idx = idx
            
        self.current_seq_idx = 0
        self.retention_counter = 0
        self.memorisation_timer = 0
        self.frames_per_color = 5 # arbitrary fixed time for flashing colours
        
        self.agent.reset()
        self._randomise_room()
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def _randomise_room(self):
        if self.randomise_wall_colors:
            if self.colour_vocabulary_size >= 4:
                wall_colors = list(self.np_random.choice(range(self.colour_vocabulary_size), size=4, replace=False))
                wall_colors = [self.vocab_colors[i] for i in wall_colors]
            else:
                wall_colors = [self.vocab_colors[self.np_random.integers(0, self.colour_vocabulary_size)] for _ in range(4)]
                
            target_color = self.sequence[self.current_seq_idx]
            if target_color not in wall_colors:
                wall_colors[self.np_random.integers(0, 4)] = target_color
        else:
            wall_colors = self.vocab_colors[:4]
            target_color = self.sequence[self.current_seq_idx]
            if target_color not in wall_colors:
                wall_colors[self.np_random.integers(0, 4)] = target_color
                
        self.wall_colors = {
            "top": wall_colors[0],
            "bottom": wall_colors[1],
            "left": wall_colors[2],
            "right": wall_colors[3]
        }

    def _get_obs(self):
        if self.current_phase == "memorisation":
            color = self.sequence[self.current_seq_idx]
            obs = np.full((self.obs_size, self.obs_size, 3), color, dtype=np.uint8)
        elif self.current_phase == "retention":
            obs = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        else:
            surf = pygame.Surface((self.obs_size, self.obs_size))
            surf.fill((0, 0, 0))
            
            scale_x = self.obs_size / self.game_size
            scale_y = self.obs_size / self.game_size
            
            rects = self.room.get_wall_rects()
            for key, rect in rects.items():
                scaled_rect = pygame.Rect(rect.x * scale_x, rect.y * scale_y, rect.width * scale_x, rect.height * scale_y)
                pygame.draw.rect(surf, self.wall_colors[key], scaled_rect)
                
            agent_rect = self.agent.get_rect()
            scaled_agent = pygame.Rect(agent_rect.x * scale_x, agent_rect.y * scale_y, agent_rect.width * scale_x, agent_rect.height * scale_y)
            pygame.draw.rect(surf, (255, 255, 255), scaled_agent)
            
            obs = pygame.surfarray.array3d(surf)
            obs = np.transpose(obs, (1, 0, 2)) # Shape is (84, 84, 3)
            
        return obs

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        
        self.step_count += 1
        
        if self.current_phase == "memorisation":
            self.memorisation_timer += 1
            if self.memorisation_timer >= self.frames_per_color:
                self.memorisation_timer = 0
                self.current_seq_idx += 1
                if self.current_seq_idx >= len(self.sequence):
                    if self.retention_delay_steps > 0:
                        self.current_phase = "retention"
                    else:
                        self.current_phase = "navigation"
                        self.agent.reset()
                        self._randomise_room()
                    self.current_seq_idx = 0 
        elif self.current_phase == "retention":
            self.retention_counter += 1
            if self.retention_counter >= self.retention_delay_steps:
                self.current_phase = "navigation"
                self.agent.reset()
                self._randomise_room()
        elif self.current_phase == "navigation":
            self.agent.move(action)
            
            agent_rect = self.agent.get_rect()
            wall_rects = self.room.get_wall_rects()
            
            hit_wall = None
            for key, rect in wall_rects.items():
                if agent_rect.colliderect(rect):
                    hit_wall = key
                    break
                    
            if hit_wall:
                touched_color = self.wall_colors[hit_wall]
                target_color = self.sequence[self.current_seq_idx]
                
                if touched_color == target_color:
                    if self.reward_density == 'dense':
                        reward += self.reward_correct_wall
                    self.current_seq_idx += 1
                    
                    if self.current_seq_idx >= len(self.sequence):
                        if self.reward_density == 'sparse':
                            reward += self.reward_sequence_completion
                        if self.infinite_mode:
                            # Start next loop immediately
                            self.current_phase = "memorisation"
                            seq_len = self.np_random.integers(self.sequence_length_range[0], self.sequence_length_range[1] + 1)
                            self.sequence = []
                            last_color_idx = -1
                            for _ in range(seq_len):
                                if self.allow_consecutive_duplicates:
                                    idx = self.np_random.integers(0, self.colour_vocabulary_size)
                                else:
                                    choices = [i for i in range(self.colour_vocabulary_size) if i != last_color_idx]
                                    idx = self.np_random.choice(choices)
                                self.sequence.append(self.vocab_colors[idx])
                                last_color_idx = idx
                            self.current_seq_idx = 0
                            self.memorisation_timer = 0
                            self.retention_counter = 0
                        else:
                            terminated = True
                    else:
                        self.agent.reset()
                        self._randomise_room()
                else:
                    self.score -= self.penalty_severity
                    reward -= self.penalty_severity
                    if self.hard_death_mode:
                        terminated = True
                    else:
                        self.agent.reset()

        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, {}
        
    def render(self):
        if self.render_mode is None:
            return
            
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Sequential Colour Navigation Task")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
            if self.icons is None:
                self.icons = {}
                self._load_icons()
            
        if self.render_mode == "human":
            self.window.fill((30, 30, 30))
            
            # Left: Game View
            game_surface = pygame.Surface((self.game_size, self.game_size))
            game_surface.fill((0, 0, 0))
            
            if self.current_phase == "memorisation":
                color = self.sequence[self.current_seq_idx]
                game_surface.fill(color)
            elif self.current_phase == "retention":
                game_surface.fill((0, 0, 0))
            else:
                rects = self.room.get_wall_rects()
                for key, rect in rects.items():
                    pygame.draw.rect(game_surface, self.wall_colors[key], rect)
                pygame.draw.rect(game_surface, (255, 255, 255), self.agent.get_rect())
                
            self.window.blit(game_surface, (0, 0))
            
            # Right: Stats Panel
            stats_x = self.game_size + 20
            
            title = self.font.render("Stats & Legend", True, (255, 255, 255))
            self.window.blit(title, (stats_x, 20))
            
            phase_text = self.font.render(f"Phase: {self.current_phase.capitalize()}", True, (200, 200, 200))
            self.window.blit(phase_text, (stats_x, 80))
            
            score_text = self.font.render(f"Score: {self.score:.1f}", True, (200, 200, 200))
            self.window.blit(score_text, (stats_x, 130))
            
            target_text = self.font.render(f"Progress: {self.current_seq_idx}/{len(self.sequence)}", True, (200, 200, 200))
            self.window.blit(target_text, (stats_x, 180))
            
            # Legend
            y_offset = 250
            legend_title = self.font.render("Legend:", True, (255, 255, 255))
            self.window.blit(legend_title, (stats_x, y_offset))
            y_offset += 50
            
            for key, label in [("agent", "Agent"), ("target_wall", "Target Wall"), ("penalty_wall", "Penalty Wall")]:
                if key in self.icons and self.icons[key]:
                    self.window.blit(self.icons[key], (stats_x, y_offset))
                lbl = self.font.render(label, True, (200, 200, 200))
                self.window.blit(lbl, (stats_x + 80, y_offset + 20))
                y_offset += 80
                
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
