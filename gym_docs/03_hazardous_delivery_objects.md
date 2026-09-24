# Hazardous Delivery Environment: Object Models and Entities

The `HazardousDeliveryEnv` environment introduces a specialized set of objects and entities that extend the foundational `minigrid` framework. These entities govern the rules of interaction (such as overlapping, taking damage, or completing objectives) and are equipped with custom visual rendering logic to make the simulation distinct and interpretable.

All object definitions for this environment can be found in the core objects module:
> 📁 [`temporal_comp_gym/envs/hazardous_delivery/objects.py`](../temporal_comp_gym/envs/hazardous_delivery/objects.py)

---

## Visual Rendering and Custom Icons

Before diving into the specific entities, it is important to understand how they are rendered. Instead of relying solely on the default Minigrid geometric shapes, the `Hazardous Delivery` environment uses a custom `CachedIcon` system to alpha-blend icons (either from the `pytablericons` package or custom images) onto the grid tiles.

### Customizing Visual Display Images
As explicitly noted in the developer `notes.txt` for this environment, you can easily swap out the default icons for your own custom images. Within the `__init__` method of any entity class, you will find an icon declaration that looks like this:

```python
self.icon = CachedIcon(pt.OutlineIcon.SKULL, "#800080", custom_path=None)
```

To change the visual display into your own custom image, simply modify the `custom_path` variable to point to your image file (e.g., `custom_path="assets/my_custom_skull.png"`). The `CachedIcon` helper class will automatically load, resize, cache, and blend that image onto the tile dynamically.

---

## Simulation Entities

Below is a breakdown of the specific objects present in the `HazardousDeliveryEnv`, detailing their properties, behaviors, and visual roles within the simulation.

### 1. Hazards

These objects pose a threat to the agent and dictate the safe paths through the gridworld.

*   **`DeathSquare`**: 
    *   **Behavior**: Acts as a fatal trap. It inherits directly from the Minigrid `Lava` class, meaning stepping on it typically terminates the episode immediately and yields a failure.
    *   **Visuals**: Renders a standard lava background with a bright orange `FLAME` icon overlaid on top of it.
*   **`DamageSquare`**: 
    *   **Behavior**: A softer hazard compared to the `DeathSquare`. It is defined as an overlapping floor tile (`can_overlap() = True`) that inflicts partial damage or a penalty to the agent over time rather than instant death.
    *   **Visuals**: Has a subtle purple background blended with a purple `SKULL` icon.

### 2. Resources

*   **`HealthNode`**: 
    *   **Behavior**: A restorative item. Agents can overlap with this floor tile to regain health or recover from damage taken by traversing through `DamageSquare` tiles.
    *   **Visuals**: Features a subtle green background blended with a bright green `HEART_PLUS` icon.

### 3. Logistics and Sequential Objects

The core task of the environment involves managing deliveries, which relies heavily on sequence IDs.

*   **`Package`**: 
    *   **Behavior**: Inherits from the `Ball` object but completely overrides the default rendering. It represents an item the agent must pick up or interact with. It can hold a specific `seq_id` (Sequence ID) to dictate the order of operations for temporal composition tasks.
    *   **Visuals**: Displays a yellow `PACKAGE` icon. If a `seq_id` is assigned to the package, a helper function (`draw_number_on_img`) overlays the sequence number directly onto the bottom-right corner of the tile for easy visual debugging.
*   **`Destination`**: 
    *   **Behavior**: Represents the drop-off point for packages. Like the package, it accepts a `seq_id` to enforce sequential delivery rules (e.g., Package 1 must specifically go to Destination 1).
    *   **Visuals**: Renders a dark blue background with a blue `TARGET` icon. It also displays its `seq_id` if one is assigned.

### 4. Level Goal

*   **`LevelGoal`**: 
    *   **Behavior**: Inherits from Minigrid's base `Goal` class. Reaching this tile typically signifies the successful completion of the current level or episode.
    *   **Visuals**: Replaces the default square Minigrid goal graphics with a green `FLAG` icon.

---

### Code Example: The Package Entity

Here is an inline snippet showing how the `Package` entity combines minigrid inheritance, custom icons, and dynamic text rendering all in one cohesive class:

```python
class Package(Ball):
    def __init__(self, color="yellow", seq_id=None):
        super().__init__(color)
        self.type = 'package'
        self.seq_id = seq_id
        
        # 1. Define the custom icon (can be overridden with custom_path)
        self.icon = CachedIcon(pt.OutlineIcon.PACKAGE, "#FFFF00", custom_path=None)
        
    def can_overlap(self):
        # 2. Allow the agent to step onto/interact with the package
        return True
        
    def render(self, img):
        # 3. Blend the package icon over a transparent floor background
        arr = self.icon.get_arr(img.shape[0])
        blend_icon(img, arr)
        
        # 4. If a sequence ID is required, draw it dynamically on the image
        if self.seq_id is not None:
            draw_number_on_img(img, str(self.seq_id))
```

By heavily customizing the rendering pipeline and extending Minigrid's `WorldObj`, the Hazardous Delivery environment provides a rich, interpretable interface for both agents and human observers.
