# Deferred Maintenance Environment: Object Models and Physics Entities

Unlike the grid-based *Hazardous Delivery* task, the **Deferred Maintenance** environment operates in a continuous, 2D physics space powered by `pymunk`. The agent is a physical body with mass, friction and momentum. The level itself is still laid out on a grid of 50-pixel cells, and each cell is a static `pymunk` body.

All physics entities for this environment are defined in:
> 📁 [`temporal_comp_gym/envs/deferred_maintenance/objects.py`](../temporal_comp_gym/envs/deferred_maintenance/objects.py)

> **Protocol v2 note.** `objects.py` itself did not change in protocol v2. What changed is how the engine uses these objects (see [`07_deferred_maintenance_core.md`](07_deferred_maintenance_core.md)): every stage is now generated from a seeded schedule with exactly one station per station type, collected code zones are replaced by path tiles, the off-path check and the local-grid sensor read a per-stage map of cell types instead of querying shapes, and the rocket's shape is excluded from sensor queries.

---

## 1. The Agent (`Rocket`)

The primary actor in this environment is the `Rocket`. It is a dynamic physical body that responds to thrust, torque and gravity.

### Physical Properties
The rocket has mass 1.0 and a 20×40 pixel box shape. It has some elasticity (bounciness, 0.3) and friction (0.5), which matter when it hits solid stage boundaries.

### Movement and Behaviors
Instead of moving between discrete grid cells, the rocket is piloted by applying continuous physical forces.
*   **`apply_thrust`**: Pushes the rocket along its own "up" axis (towards its nose), so thrust direction follows the rocket's current tilt.
*   **`apply_rotation`**: Sets the torque on the body, spinning it around its centre of mass.

How actions map to these calls (thrust scale, air brakes, auto-levelling, the ±90° tilt limit) is handled by the core engine.

```python
class Rocket:
    def __init__(self, space, position, mass=1.0):
        self.width = 20
        self.height = 40
        moment = pymunk.moment_for_box(mass, (self.width, self.height))
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        
        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))
        self.shape.elasticity = 0.3
        self.shape.friction = 0.5
        self.shape.collision_type = 1 # Rocket
        
        space.add(self.body, self.shape)
        
    def apply_thrust(self, force_magnitude):
        self.body.apply_force_at_local_point((0, -force_magnitude), (0, 0))
        
    def apply_rotation(self, torque):
        self.body.torque = torque
```

After creating the rocket, the engine puts its shape in its own query category, so sensor queries (raycasts) never hit the rocket itself.

---

## 2. Static Zones and Objectives

The rest of the environment is made of static, cell-sized bodies. All of them except a solid `StageBoundary` are **sensors** (`self.shape.sensor = True`): the rocket can overlap them without any physical bounce. The engine detects overlaps by querying the rocket's shape against the space on every step and then applies the task rules.

All static zones are created from a `rect` parameter `(x, y, width, height)` that defines their cell.

### Logic-Driven Zones (`CodeZone` & `MaintenanceStation`)
These are the core objects of the task. Each carries a `color_code` (its type, `1` to `max_code_types`).

*   **`CodeZone`**: A maintenance code to collect. The rocket collects it the first time any part of the rocket overlaps it (no hovering needed). At the end of that step the zone is removed and replaced by a plain `PathBoundary` tile. Codes appear only in the stages before readiness, one distinct type per stage at the default settings.
*   **`MaintenanceStation`**: Where a held code is serviced. The rocket must overlap the station continuously for `required_hover_duration` seconds (2.0 s = 120 steps by default). Stations are never removed. Every stage contains exactly one station of each station type; with `distractor_stations=True` (the protocol v2 main condition) that is all `max_code_types` types, so the stations on screen never reveal which codes the rocket holds.

```python
class CodeZone:
    def __init__(self, space, rect, color_code):
        x, y, w, h = rect
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (x + w/2, y + h/2)
        self.shape = pymunk.Poly.create_box(self.body, (w, h))
        self.shape.sensor = True
        self.shape.collision_type = 3 # Code Zone
        self.color_code = color_code
        space.add(self.body, self.shape)
        self.rect = rect

class MaintenanceStation:
    def __init__(self, space, rect, color_code):
        x, y, w, h = rect
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (x + w/2, y + h/2)
        self.shape = pymunk.Poly.create_box(self.body, (w, h))
        self.shape.sensor = True
        self.shape.collision_type = 4 # Station
        self.color_code = color_code
        space.add(self.body, self.shape)
        self.rect = rect
```

Code zones and stations are always placed at least two cells apart (Chebyshev distance) from each other and from the exit, so the rocket can never touch two of them, or one of them and the exit, at the same time. They can be next to the entry.

### Stage Navigation (`StageEntry`, `StageExit`, & `PathBoundary`)
These zones manage the spatial flow of the agent.

*   **`StageEntry`**: The first cell of the stage's path. The rocket spawns at its centre at the start of every stage, with zero velocity and an upright angle.
*   **`StageExit`**: The last cell of the path. Touching it ends the stage: the rocket moves on to the next stage, or the episode ends (on the last stage, or if a code in this stage was left uncollected).
*   **`PathBoundary`**: An ordinary path cell (the "safe" route between entry and exit). Leaving the path for an empty (void) cell costs `-0.1` per step; it never ends the episode.

```python
class StageExit:
    def __init__(self, space, rect):
        x, y, w, h = rect
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (x + w/2, y + h/2)
        self.shape = pymunk.Poly.create_box(self.body, (w, h))
        self.shape.sensor = True
        self.shape.collision_type = 5 # Exit
        space.add(self.body, self.shape)
        self.rect = rect
```

`StageEntry` and `PathBoundary` are built the same way, with collision types 6 and 2.

### Physical Walls (`StageBoundary`)
A ring of `StageBoundary` cells surrounds every stage. With `enforce_boundaries=True` (the default) they are solid walls the rocket bounces off; with `False` they are sensors the rocket can pass through. Either way, every step on which the rocket overlaps a boundary costs `-5`.

```python
class StageBoundary:
    def __init__(self, space, rect, is_solid=True):
        x, y, w, h = rect
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (x + w/2, y + h/2)
        self.shape = pymunk.Poly.create_box(self.body, (w, h))
        self.shape.sensor = not is_solid
        self.shape.collision_type = 7 # Boundary
        space.add(self.body, self.shape)
        self.rect = rect
```

---

## 3. Collision Type and Cell Type Reference

The engine does not register `pymunk` collision callbacks; it identifies overlapping objects by their Python class. The integer collision types now serve a second purpose: they are the **cell-type codes** that the engine stores for every cell of a stage and that the `local_grid` sensor reports (see [`08_deferred_maintenance_wrappers.md`](08_deferred_maintenance_wrappers.md)).

| Code | Object | Appears in the local-grid sensor? |
|---|---|---|
| `0` | Empty void (no object) | yes |
| `1` | Rocket (the agent) | no (only objects of the level are mapped) |
| `2` | Path Boundary (safe path, including collected code zones) | yes |
| `3` | Code Zone (not yet collected) | yes |
| `4` | Maintenance Station | yes |
| `5` | Stage Exit | yes |
| `6` | Stage Entry | yes |
| `7` | Stage Boundary (walls) | yes |
