# Deferred Maintenance Environment: Object Models and Physics Entities

Unlike the grid-based *Hazardous Delivery* task, the **Deferred Maintenance** environment operates in a continuous, 2D physics space powered by `pymunk`. Consequently, the objects in this environment are not tile-based grid entities, but rather physical bodies with properties like mass, friction, momentum, and continuous coordinate tracking.

All physics entities for this environment are defined in:
> 📁 [`temporal_comp_gym/envs/deferred_maintenance/objects.py`](../temporal_comp_gym/envs/deferred_maintenance/objects.py)

---

## 1. The Agent (`Rocket`)

The primary actor in this environment is the `Rocket`. It is a dynamic physical body that responds to external forces, gravity, and torques.

### Physical Properties
The rocket is instantiated with a specific mass and size (20x40 bounding box). It possesses basic elasticity (bounciness) and friction to interact realistically with solid walls and boundaries.

### Movement and Behaviors
Instead of teleporting between discrete grid cells, the rocket is piloted by applying continuous physical forces. 
*   **`apply_thrust`**: Pushes the rocket "forward" relative to its current local orientation.
*   **`apply_rotation`**: Spins the rocket around its center of mass by applying torque.

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
        self.shape.collision_type = 1 # Collision ID for Rocket
        
        space.add(self.body, self.shape)
        
    def apply_thrust(self, force_magnitude):
        # Applies upward force in the local coordinate system
        self.body.apply_force_at_local_point((0, -force_magnitude), (0, 0))
        
    def apply_rotation(self, torque):
        # Spins the body
        self.body.torque = torque
```

---

## 2. Static Zones and Objectives

The rest of the environment is composed of static bodies—entities that do not move but dictate the spatial rules and objectives of the level. Most of these objects are configured as **sensors** (`self.shape.sensor = True`). A sensor shape detects when the `Rocket` overlaps with it (triggering game logic and rewards in the core engine) but does not cause a physical bounce or collision response.

All static zones are instantiated using a `rect` parameter `(x, y, width, height)` to define their bounding box.

### Logic-Driven Zones (`CodeZone` & `MaintenanceStation`)
These are the core objects driving the temporal composition task. The agent must visit specific zones based on matching `color_code` properties to pick up maintenance codes and deliver them to their respective stations.

```python
class CodeZone:
    def __init__(self, space, rect, color_code):
        x, y, w, h = rect
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (x + w/2, y + h/2)
        
        self.shape = pymunk.Poly.create_box(self.body, (w, h))
        self.shape.sensor = True          # Detects overlap without physical collision
        self.shape.collision_type = 3     # Collision ID for Code Zone
        
        self.color_code = color_code      # Crucial for sequential logic matching
        space.add(self.body, self.shape)
        self.rect = rect

class MaintenanceStation:
    def __init__(self, space, rect, color_code):
        # ... structurally identical to CodeZone ...
        self.shape.collision_type = 4     # Collision ID for Station
        self.color_code = color_code
        # ...
```

### Stage Navigation (`StageEntry`, `StageExit`, & `PathBoundary`)
These zones manage the spatial flow of the agent.

*   **`StageEntry`**: Defines the designated starting area or spawn point for the rocket.
*   **`StageExit`**: Represents the goal area the agent must reach to complete the level.
*   **`PathBoundary`**: Represents a designated "safe zone" or required pathway. Straying outside these zones likely results in penalties or episode termination.

```python
class StageExit:
    def __init__(self, space, rect):
        x, y, w, h = rect
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (x + w/2, y + h/2)
        
        self.shape = pymunk.Poly.create_box(self.body, (w, h))
        self.shape.sensor = True
        self.shape.collision_type = 5 # Collision ID for Exit
        # ...
```

### Physical Walls (`StageBoundary`)
Unlike the sensors, the `StageBoundary` can act as a solid, physical wall to keep the rocket contained within the play area.

```python
class StageBoundary:
    def __init__(self, space, rect, is_solid=True):
        # ...
        # If is_solid is True, sensor becomes False, making it a physical barrier
        self.shape.sensor = not is_solid 
        self.shape.collision_type = 7 # Collision ID for Boundary
        # ...
```

---

## 3. Collision Type Mapping Reference

To process game logic efficiently, `pymunk` relies on integer collision types to route collision callbacks (e.g., instructing the engine what to do when object Type 1 touches object Type 5). Here is the definitive mapping used across the Deferred Maintenance environment:

*   **`1`** - Rocket (The Agent)
*   **`2`** - Path Boundary (Safe Zone)
*   **`3`** - Code Zone (Task Pickup)
*   **`4`** - Maintenance Station (Task Dropoff / Action)
*   **`5`** - Stage Exit (Level Goal)
*   **`6`** - Stage Entry (Spawn Point)
*   **`7`** - Stage Boundary (Walls)
