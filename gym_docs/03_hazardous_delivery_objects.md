# Hazardous Delivery Environment: Object Models and Entities

The `HazardousDeliveryEnv` environment introduces a specialized set of objects and entities that extend the foundational `minigrid` framework. These entities define what the agent can stand on and how they are drawn. The rules that fire when the agent enters each one (rewards, health, termination) live in the core engine (see [`04_hazardous_delivery_core.md`](04_hazardous_delivery_core.md)); this page summarises them so each object's role is clear.

All object definitions for this environment can be found in the core objects module:
> 📁 [`temporal_comp_gym/envs/hazardous_delivery/objects.py`](../temporal_comp_gym/envs/hazardous_delivery/objects.py)

> **Protocol v2 note.** `objects.py` itself did not change in protocol v2. What changed is how the engine uses it: ordered deliveries (`task_dependencies=True`) are no longer supported and raise an error, so packages and destinations are never numbered and the sequence-number overlay is never drawn. Every package is interchangeable with every other package, and any destination accepts any package.

---

## Visual Rendering and Custom Icons

Before diving into the specific entities, it is important to understand how they are rendered. Instead of relying solely on the default Minigrid geometric shapes, the `Hazardous Delivery` environment uses a custom `CachedIcon` system to alpha-blend icons (either from the `pytablericons` package or custom images) onto the grid tiles. Rendering only affects the human/RGB view; the agent never sees these images (its observation is the vector described in [`05_hazardous_delivery_wrappers.md`](05_hazardous_delivery_wrappers.md)).

### Customizing Visual Display Images
As noted in the developer `notes.txt` for this environment, you can swap out the default icons for your own custom images. Within the `__init__` method of any entity class, you will find an icon declaration that looks like this:

```python
self.icon = CachedIcon(pt.OutlineIcon.SKULL, "#800080", custom_path=None)
```

To change the visual display into your own custom image, modify the `custom_path` variable to point to your image file (e.g., `custom_path="assets/my_custom_skull.png"`). The `CachedIcon` helper class loads, resizes, caches (per tile size) and blends that image onto the tile. If the path is `None` or the file does not exist, it falls back to the `pytablericons` icon.

---

## Simulation Entities

Below is a breakdown of the objects present in the `HazardousDeliveryEnv`, detailing their properties, behaviour and visual roles. The "sensor value" is what the agent's proximity sensors report for that object.

| Object | `type` | Agent can enter? | Sensor value | Removed after use? |
|---|---|---|---|---|
| Empty floor | – | yes | `0.0` | – |
| Wall (outer border) | `wall` | no | `0.0` | – |
| `DeathSquare` | `lava` | yes (fatal) | `2.0` | – (episode ends) |
| `DamageSquare` | `damage` | yes | `1.0` | no |
| `HealthNode` | `health` | yes | `0.0` | yes |
| `Package` | `package` | yes | `3.0` | only when picked up |
| `Destination` | `destination` | yes | `3.0` | only when a delivery happens |
| `LevelGoal` | `goal` | yes (ends episode) | `4.0` | – (episode ends) |

### 1. Hazards

These objects pose a threat to the agent and dictate the safe paths through the gridworld.

*   **`DeathSquare`**:
    *   **Behavior**: A fatal trap. It inherits from the Minigrid `Lava` class (so the agent can step onto it). Entering it ends the episode immediately (`terminated=True`) with a reward of `-50` and termination cause `lava`.
    *   **Visuals**: Renders a standard lava background with a bright orange `FLAME` icon overlaid on top of it.
*   **`DamageSquare`**:
    *   **Behavior**: A softer hazard. It is an overlapping floor tile (`can_overlap() = True`). Each step that ends on it costs `-1` reward and 1 health point. If health reaches 0 on that step, the episode ends with an additional `-50` (total `-51`) and cause `damage_death`. The square is not consumed, so standing on it and bumping into a wall hurts again.
    *   **Visuals**: Has a subtle purple background blended with a purple `SKULL` icon.

### 2. Resources

*   **`HealthNode`**:
    *   **Behavior**: A single-use restorative tile. Entering it restores health to the maximum (10) and removes the node from the grid. It gives no reward. Note that the agent's sensors report health nodes as `0.0`, the same as empty floor, so the agent cannot see them coming.
    *   **Visuals**: Features a subtle green background blended with a bright green `HEART_PLUS` icon.

### 3. Logistics Objects

The core task of the environment involves picking up packages and dropping them at destinations.

*   **`Package`**:
    *   **Behavior**: Inherits from the `Ball` object but completely overrides the default rendering. Entering it picks it up if the agent is carrying fewer packages than its capacity (1 by default); the package is then removed from the grid. If the agent is already full, the package stays where it is. The class still accepts a `seq_id` argument, but the engine always passes `None` in protocol v2.
    *   **Visuals**: Displays a yellow `PACKAGE` icon. The `draw_number_on_img` overlay only runs when `seq_id` is set, which no longer happens.
*   **`Destination`**:
    *   **Behavior**: The drop-off point. Entering it while carrying at least one package delivers one package (`+10` reward) and removes the destination from the grid. Entering it empty-handed does nothing and the destination stays. Any destination accepts any package.
    *   **Visuals**: Renders a dark blue background with a blue `TARGET` icon.

Packages and destinations share the same sensor value (`3.0`), so from the observation alone the agent cannot tell which one it is standing next to; it has to remember whether it is carrying something.

### 4. Level Goal

*   **`LevelGoal`**:
    *   **Behavior**: Inherits from Minigrid's base `Goal` class. Entering it always ends the episode. With every package delivered it gives `+50` and counts as success; otherwise it gives `-50` with cause `premature_goal`.
    *   **Visuals**: Replaces the default square Minigrid goal graphics with a green `FLAG` icon.

### Minigrid Encodings

`DamageSquare` and `HealthNode` encode themselves as Minigrid `floor` objects, `Package` as a `ball` and `Destination` as a `box`. These encodings only matter for Minigrid's built-in image observation and for `grid.encode()` (used in tests to compare layouts); the agent's vector observation does not use them.

---

### Code Example: The Package Entity

Here is the current `Package` class, showing how it combines Minigrid inheritance, a custom icon, and the (now unused) sequence-number overlay:

```python
class Package(Ball):
    def __init__(self, color="yellow", seq_id=None):
        super().__init__(color)
        self.type = 'package'
        self.seq_id = seq_id
        # Custom path can be set here:
        self.icon = CachedIcon(pt.OutlineIcon.PACKAGE, "#FFFF00", custom_path=None)
        
    def can_overlap(self):
        return True
        
    def render(self, img):
        # We don't call super() because we don't want the basic circle shape.
        # We just want the icon over a transparent/floor background.
        arr = self.icon.get_arr(img.shape[0])
        blend_icon(img, arr)
        if self.seq_id is not None:
            draw_number_on_img(img, str(self.seq_id))
        
    def encode(self):
        return (OBJECT_TO_IDX["ball"], COLOR_TO_IDX[self.color], 0)
```

By customizing the rendering pipeline and extending Minigrid's `WorldObj`, the Hazardous Delivery environment provides an interpretable view for human observers, while the agent itself works from a compact vector observation.
