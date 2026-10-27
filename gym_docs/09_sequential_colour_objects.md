# Sequential Colour Environment: Object Models and Entities

Unlike the previous environments, the **Sequential Colour** environment does not rely on external spatial frameworks like `minigrid` (array-based gridworld) or `pymunk` (rigid-body physics). Instead, it implements a very lightweight, bespoke 2D spatial engine built around `pygame`.

Entities in this environment move in pixel coordinates and handle collisions using simple Axis-Aligned Bounding Box (AABB) checks via `pygame.Rect`.

The core entity definitions are located in:
> 📁 [`temporal_comp_gym/envs/sequential_colour/objects.py`](../temporal_comp_gym/envs/sequential_colour/objects.py)

> **Protocol v2 note.** `objects.py` did not change in protocol v2, and neither did the room geometry or movement. The changes to this task are in the core engine (events, an episode limit, a configurable cue duration) and the noise wrapper; see [`10_sequential_colour_core.md`](10_sequential_colour_core.md) and [`11_sequential_colour_wrappers.md`](11_sequential_colour_wrappers.md).

---

## 1. The `Agent`

The `Agent` class represents the actor navigating the room. It tracks its own coordinates, with no external physics engine.

### Properties and Initialization
The agent is initialized with a starting position, a bounding box `size`, and a movement `speed`. The class defaults (`size=10`, `speed=5`) are not used: the core engine always creates the agent at the centre of the room with `size` equal to one block and `speed` equal to half a block (see "Room Scale" below).

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
        self.x = self.start_x
        self.y = self.start_y
```

`reset()` puts the agent back at the centre. The engine calls it at the start of navigation, after every wall touch (correct or wrong), and on every episode reset.

### Movement Logic
The environment takes discrete action inputs (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT). Each action shifts the agent's pixel coordinates by `speed`. There is no velocity or momentum: the agent moves exactly one `speed` step per action, and only during the navigation phase.

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
The agent exposes its footprint to the core engine as a `pygame.Rect` centred on its current `x` and `y`. The engine checks this rectangle against the four wall rectangles after every move.

```python
    def get_rect(self):
        return pygame.Rect(self.x - self.size/2, self.y - self.size/2, self.size, self.size)
```

`pygame.Rect` stores whole pixels, so the fractional coordinates are truncated when the rectangle is built.

---

## 2. The `Room` (Boundaries)

The `Room` class defines the play area and its four walls. It returns the walls (Top, Bottom, Left, Right) as a dictionary of `pygame.Rect` objects. The top and bottom walls span the full width; the left and right walls fill the space between them.

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

The walls do not physically block the agent. Instead, the first overlap with a wall counts as a "touch": the engine scores it and puts the agent back in the centre, so the agent never passes through a wall. The `Room` does not store wall colours; the engine assigns a colour to each wall key (`self.wall_colors`).

---

## 3. Room Scale

The core engine sizes both objects from one setting, `agent_wall_distance_blocks` (default 6):

*   The room is 800 × 800 pixels and is divided into `2 × distance + 3` blocks per side (15 at the default), so one block is about 53.3 pixels.
*   Walls are one block thick, and the agent is one block wide.
*   The agent moves half a block per action, so it covers one block every two steps.
*   With the default distance of 6 blocks, the agent touches any wall on the **13th** step of moving straight towards it from the centre.

In the agent's 84 × 84 observation the room is scaled down by 84/800, so walls are about 5 pixels thick and the agent is a white square of about 5 pixels.

*(Note: The `Agent` and `Room` are the spatial foundation; the wall colours, the colour sequence and all scoring are handled in the environment's `core.py` engine.)*
