# Sequential Colour Environment: Object Models and Entities

Unlike the previous environments, the **Sequential Colour** environment does not rely on external spatial frameworks like `minigrid` (array-based gridworld) or `pymunk` (complex rigid-body physics). Instead, it implements a very lightweight, bespoke 2D spatial engine built entirely around `pygame`.

Entities in this environment move across continuous pixel coordinates and handle collisions using simple Axis-Aligned Bounding Box (AABB) mathematics via `pygame.Rect`.

The core entity definitions are located in:
> 📁 [`temporal_comp_gym/envs/sequential_colour/objects.py`](../temporal_comp_gym/envs/sequential_colour/objects.py)

---

## 1. The `Agent`

The `Agent` class represents the player or AI actor navigating the space. It tracks its own coordinates and velocity natively, abstracting away the need for an external physics engine.

### Properties and Initialization
The agent is initialized with a starting position, a bounding box `size`, and a discrete movement `speed`.

```python
class Agent:
    def __init__(self, x, y, size=10, speed=5):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.size = size
        self.speed = speed
        
    def reset(self):
        # Cleanly resets the agent to its spawn point
        self.x = self.start_x
        self.y = self.start_y
```

### Movement Logic
The environment takes discrete action inputs (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT). However, instead of shifting the agent by one grid tile at a time, it shifts the agent's absolute pixel coordinates dynamically by the `speed` multiplier. 

```python
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
```

### Collision Detection
Because there is no external physics engine, the agent must expose its spatial footprint to the core environment engine. It does this by generating a `pygame.Rect` centered precisely on its current `x` and `y` coordinates. The main game loop will use this rectangle to calculate overlaps with targets and walls.

```python
    def get_rect(self):
        # Returns a rectangle centered on the agent for AABB collision checks
        return pygame.Rect(self.x - self.size/2, self.y - self.size/2, self.size, self.size)
```

---

## 2. The `Room` (Boundaries)

The `Room` class acts as the spatial container for the environment. It defines the outer bounds of the play area and generates the physical walls that restrict the agent's movement.

### Wall Generation
The room uses its `width`, `height`, and `wall_thickness` to mathematically construct four solid boundaries (Top, Bottom, Left, and Right). These are returned as a dictionary of `pygame.Rect` objects, which the core engine will check against the agent's rectangle to prevent out-of-bounds movement.

```python
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
```

*(Note: While the `Agent` and `Room` form the spatial foundation, the interactive colored zones that drive the sequential task logic are handled dynamically in the environment's `core.py` engine.)*
