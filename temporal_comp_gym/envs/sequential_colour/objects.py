import pygame

class Agent:
    def __init__(self, x, y, size=10, speed=5):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.size = size
        self.speed = speed
        
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y

    def move(self, action):
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        if action == 0:
            self.y -= self.speed
        elif action == 1:
            self.y += self.speed
        elif action == 2:
            self.x -= self.speed
        elif action == 3:
            self.x += self.speed
            
    def get_rect(self):
        return pygame.Rect(self.x - self.size/2, self.y - self.size/2, self.size, self.size)

class Room:
    def __init__(self, width, height, wall_thickness=20):
        self.width = width
        self.height = height
        self.wall_thickness = wall_thickness

    def get_wall_rects(self):
        return {
            "top": pygame.Rect(0, 0, self.width, self.wall_thickness),
            "bottom": pygame.Rect(0, self.height - self.wall_thickness, self.width, self.wall_thickness),
            "left": pygame.Rect(0, self.wall_thickness, self.wall_thickness, self.height - 2*self.wall_thickness),
            "right": pygame.Rect(self.width - self.wall_thickness, self.wall_thickness, self.wall_thickness, self.height - 2*self.wall_thickness)
        }
