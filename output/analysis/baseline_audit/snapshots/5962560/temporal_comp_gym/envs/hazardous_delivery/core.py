import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from .objects import DeathSquare, DamageSquare, HealthNode, Package, Destination, LevelGoal, CachedIcon, blend_icon
import pytablericons as pt
class HazardousDeliveryCoreEnv(MiniGridEnv):
    def __init__(
        self,
        grid_size=10, # The width and height of the 2D grid
        percent_death_squares=0.05, # The percentage of the grid covered by instant-kill death squares
        percent_damage_squares=0.05, # The percentage of the grid covered by damage-inducing squares
        randomise_start=True, # If True, the agent spawns at a random safe coordinate each episode
        randomise_goal=True, # If True, the location of the Goal square changes every episode
        num_packages=3, # The total number of packages that must be delivered to unlock the goal square
        carrying_capacity=1, # The maximum number of packages the agent can hold simultaneously
        health_regeneration_nodes=2, # The number of single-use healing squares spawned on the grid
        task_dependencies=False, # If True, packages must be collected and delivered in a specific sequential order
        
        # Rewards and Penalty ====================
        reward_delivery=10.0, # Positive reward granted when a package is deposited at its correct destination
        reward_task_completion=50.0, # Large positive terminal reward for safely entering the Goal square after all deliveries
        penalty_damage=-1.0, # Negative reward applied each time the agent steps on a damage square
        penalty_premature_goal=-50.0, # Large negative terminal reward for entering the Goal square before deliveries are complete
        penalty_death_square=-50.0, # Large negative terminal reward for stepping into a death square hazard
        max_steps=500,
        **kwargs
    ):
        self.grid_size = grid_size
        self.percent_death_squares = percent_death_squares
        self.percent_damage_squares = percent_damage_squares
        self.randomise_start = randomise_start
        self.randomise_goal = randomise_goal
        self.num_packages = num_packages
        self.carrying_capacity = carrying_capacity
        self.health_regeneration_nodes = health_regeneration_nodes
        self.task_dependencies = task_dependencies
        
        self.reward_delivery = reward_delivery
        self.reward_task_completion = reward_task_completion
        self.penalty_damage = penalty_damage
        self.penalty_premature_goal = penalty_premature_goal
        self.penalty_death_square = penalty_death_square

        self.max_health = 10
        self.health = self.max_health
        self.carried_packages = 0
        self.delivered_packages = 0
        
        mission_space = MissionSpace(mission_func=lambda: "Deliver all packages and reach the goal.")
        
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            **kwargs
        )
        
        # Override action space to discrete directional: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = gym.spaces.Discrete(4)
        
        # Agent custom icon
        # Custom path can be set here:
        self.agent_icon = CachedIcon(pt.OutlineIcon.TRUCK_DELIVERY, "#FFFFFF", custom_path=None)
        
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        num_inner = (width - 2) * (height - 2)
        num_death = int(num_inner * self.percent_death_squares)
        num_damage = int(num_inner * self.percent_damage_squares)
        
        if self.randomise_goal:
            self.place_obj(LevelGoal())
        else:
            self.put_obj(LevelGoal(), width - 2, height - 2)
            
        for i in range(self.num_packages):
            seq_id = (i + 1) if self.task_dependencies else None
            self.place_obj(Package(seq_id=seq_id))
            
        for i in range(self.num_packages):
            seq_id = (i + 1) if self.task_dependencies else None
            self.place_obj(Destination(seq_id=seq_id))
            
        for _ in range(num_death):
            self.place_obj(DeathSquare())
            
        for _ in range(num_damage):
            self.place_obj(DamageSquare())
            
        for _ in range(self.health_regeneration_nodes):
            self.place_obj(HealthNode())
            
        if self.randomise_start:
            self.place_agent()
        else:
            self.agent_pos = (1, 1)
            self.agent_dir = 0
            
    def step(self, action):
        self.step_count += 1
        
        reward = 0.0
        terminated = False
        truncated = False
        
        x, y = self.agent_pos
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
            
        fwd_cell = self.grid.get(x, y)
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = (x, y)
            
        curr_cell = self.grid.get(self.agent_pos[0], self.agent_pos[1])
        
        if curr_cell is not None:
            if curr_cell.type == 'lava':
                reward = self.penalty_death_square
                terminated = True
                
            elif curr_cell.type == 'damage':
                reward = self.penalty_damage
                self.health -= 1
                if self.health <= 0:
                    terminated = True
                    reward += self.penalty_death_square
                    
            elif curr_cell.type == 'health':
                self.health = self.max_health
                self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
                
            elif curr_cell.type == 'package':
                # Check sequential order if task_dependencies is enabled
                can_pickup = True
                if self.task_dependencies:
                    # The next required package sequence number is (carried + delivered + 1)
                    if curr_cell.seq_id != (self.carried_packages + self.delivered_packages + 1):
                        can_pickup = False
                        
                if can_pickup and self.carried_packages < self.carrying_capacity:
                    self.carried_packages += 1
                    self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
                    
            elif curr_cell.type == 'destination':
                can_deliver = True
                if self.task_dependencies:
                    # The next required destination sequence number is (delivered + 1)
                    if curr_cell.seq_id != (self.delivered_packages + 1):
                        can_deliver = False
                        
                if can_deliver and self.carried_packages > 0:
                    self.carried_packages -= 1
                    self.delivered_packages += 1
                    reward = self.reward_delivery
                    self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
                    
            elif curr_cell.type == 'goal':
                if self.delivered_packages >= self.num_packages:
                    reward = self.reward_task_completion
                    terminated = True
                else:
                    reward = self.penalty_premature_goal
                    terminated = True
                    
        if self.step_count >= self.max_steps:
            truncated = True
            
        obs = self.gen_obs()
        info = {}
        
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.health = self.max_health
        self.carried_packages = 0
        self.delivered_packages = 0
        return super().reset(seed=seed, options=options)

    def append_legend(self, img, tile_size):
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        height = img.shape[0]
        legend_width = max(250, tile_size * 6)
        legend_img = Image.new("RGB", (legend_width, height), (30, 30, 30))
        draw = ImageDraw.Draw(legend_img)
        
        try:
            # Try to use Arial, fallback to default if not found
            font_large = ImageFont.truetype("arial.ttf", size=max(16, int(tile_size * 0.6)))
            font_small = ImageFont.truetype("arial.ttf", size=max(14, int(tile_size * 0.45)))
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
            
        # Title
        draw.text((15, 15), "Hazardous Delivery", fill=(255, 255, 255), font=font_large)
        
        # Dynamic State
        state_y = 15 + tile_size
        line_height = max(25, int(tile_size * 0.6))
        
        draw.text((15, state_y), f"Health: {self.health} / {self.max_health}", fill=(255, 100, 100), font=font_small)
        draw.text((15, state_y + line_height), f"Carrying: {self.carried_packages} / {self.carrying_capacity}", fill=(255, 255, 100), font=font_small)
        draw.text((15, state_y + line_height * 2), f"Delivered: {self.delivered_packages} / {self.num_packages}", fill=(100, 255, 100), font=font_small)
        
        # Draw a horizontal separator line
        line_y = state_y + line_height * 3 + 10
        draw.line([(15, line_y), (legend_width - 15, line_y)], fill=(100, 100, 100), width=2)
        
        # Gather legend items
        items = [("Agent", self.agent_icon)]
        
        # Instantiate dummy objects to read their configured icons
        if not hasattr(self, '_dummy_objects'):
            self._dummy_objects = [
                ("Death Square", DeathSquare()),
                ("Damage Square", DamageSquare()),
                ("Health Node", HealthNode()),
                ("Package", Package()),
                ("Destination", Destination()),
                ("Level Goal", LevelGoal()),
            ]
            
        for name, obj in self._dummy_objects:
            if hasattr(obj, 'icon'):
                items.append((name, obj.icon))
        
        # Draw items
        start_y = line_y + 20
        icon_size = int(tile_size * 0.8)
        
        for i, (name, icon_obj) in enumerate(items):
            y = start_y + i * (icon_size + 15)
            if y + icon_size > height:
                break # Avoid drawing outside the panel if window is too small
                
            icon_arr = icon_obj.get_arr(icon_size)
            if icon_arr is not None:
                icon_pil = Image.fromarray(icon_arr)
                legend_img.paste(icon_pil, (15, y), icon_pil)
                
            draw.text((25 + icon_size, y + icon_size//4), name, fill=(200, 200, 200), font=font_small)
            
        legend_arr = np.array(legend_img)
        # Concatenate horizontally
        return np.concatenate((img, legend_arr), axis=1)

    def get_full_render(self, highlight, tile_size):
        saved_pos = self.agent_pos
        self.agent_pos = (-1, -1)
        
        try:
            img = super().get_full_render(highlight, tile_size)
        finally:
            self.agent_pos = saved_pos
            
        if self.agent_pos is not None:
            x, y = self.agent_pos
            ymin = y * tile_size
            ymax = (y + 1) * tile_size
            xmin = x * tile_size
            xmax = (x + 1) * tile_size
            
            arr = self.agent_icon.get_arr(tile_size)
            tile_img = img[ymin:ymax, xmin:xmax, :]
            blend_icon(tile_img, arr)
            
        return self.append_legend(img, tile_size)

    def get_pov_render(self, tile_size):
        saved_pos = self.agent_pos
        self.agent_pos = (-1, -1)
        
        try:
            img = super().get_pov_render(tile_size)
        finally:
            self.agent_pos = saved_pos
            
        if self.agent_pos is not None:
            agent_view_size = self.agent_view_size
            view_x = agent_view_size // 2
            view_y = agent_view_size - 1
            
            ymin = view_y * tile_size
            ymax = (view_y + 1) * tile_size
            xmin = view_x * tile_size
            xmax = (view_x + 1) * tile_size
            
            arr = self.agent_icon.get_arr(tile_size)
            tile_img = img[ymin:ymax, xmin:xmax, :]
            blend_icon(tile_img, arr)
            
        return self.append_legend(img, tile_size)

    def render(self):
        import pygame
        import numpy as np

        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            img_width, img_height = img.shape[:2]
            
            if self.window is None:
                pygame.init()
                pygame.display.init()
                
                # Maintain aspect ratio based on screen_size (usually 640)
                scale = self.screen_size / img_height
                self.render_width = int(img_width * scale)
                self.render_height = int(img_height * scale)
                
                self.window = pygame.display.set_mode((self.render_width, self.render_height))
                pygame.display.set_caption("Hazardous Delivery")
                
            if self.clock is None:
                self.clock = pygame.time.Clock()
                
            surf = pygame.surfarray.make_surface(img)
            bg = pygame.transform.smoothscale(surf, (self.render_width, self.render_height))

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img
