from minigrid.core.world_object import WorldObj, Lava, Goal, Ball
from minigrid.utils.rendering import fill_coords, point_in_rect
from minigrid.core.constants import COLORS, OBJECT_TO_IDX, COLOR_TO_IDX
import numpy as np
from PIL import Image
import pytablericons as pt
import os

class CachedIcon:
    """
    Helper class to manage loading icons from paths or pytablericons.
    It caches the NumPy arrays of the icons dynamically based on rendering size.
    """
    def __init__(self, icon_enum, fallback_color, custom_path=None):
        self.icon_enum = icon_enum
        self.fallback_color = fallback_color
        # ---------------------------------------------------------
        # INSERT CUSTOM IMAGE PATH HERE:
        # e.g., custom_path = "assets/my_custom_icon.png"
        # If this is None or the file doesn't exist, it defaults
        # to the high-quality TablerIcons from the pytablericons package.
        # ---------------------------------------------------------
        self.custom_path = custom_path
        self._cache = {}

    def get_arr(self, size):
        if size in self._cache:
            return self._cache[size]
            
        if self.custom_path is not None and os.path.exists(self.custom_path):
            try:
                img = Image.open(self.custom_path).convert("RGBA").resize((size, size))
                arr = np.array(img)
                self._cache[size] = arr
                return arr
            except Exception as e:
                print(f"Failed to load custom icon at {self.custom_path}: {e}")
                
        # Fallback to pytablericons
        try:
            pil_img = pt.TablerIcons.load(self.icon_enum, size=size, color=self.fallback_color)
            arr = np.array(pil_img)
            self._cache[size] = arr
            return arr
        except Exception as e:
            print(f"Failed to load fallback icon: {e}")
            return None

def blend_icon(base_img, icon_arr):
    """Alpha-blends a loaded icon onto the existing tile image."""
    if icon_arr is None:
        return
    
    target_size = (base_img.shape[1], base_img.shape[0])
    if (icon_arr.shape[1], icon_arr.shape[0]) != target_size:
        icon_img = Image.fromarray(icon_arr).resize(target_size)
        icon_arr = np.array(icon_img)

    if icon_arr.shape[-1] == 4: # RGBA Alpha blending
        alpha = icon_arr[:, :, 3] / 255.0
        for c in range(3):
            base_img[:, :, c] = (alpha * icon_arr[:, :, c] + (1 - alpha) * base_img[:, :, c])
    else:
        base_img[:, :, :] = icon_arr[:, :, :3]

class DeathSquare(Lava):
    def __init__(self):
        super().__init__()
        self.type = 'lava'
        # Custom path can be set here:
        self.icon = CachedIcon(pt.OutlineIcon.FLAME, "#FF4500", custom_path=None)

    def render(self, img):
        # We can draw the standard Lava background and alpha-blend our icon over it
        super().render(img)
        arr = self.icon.get_arr(img.shape[0])
        blend_icon(img, arr)


class DamageSquare(WorldObj):
    def __init__(self):
        super().__init__("floor", "purple")
        self.type = "damage"
        # Custom path can be set here:
        self.icon = CachedIcon(pt.OutlineIcon.SKULL, "#800080", custom_path=None)
        
    def can_overlap(self):
        return True
        
    def render(self, img):
        # Fill a subtle background before blending the icon
        fill_coords(img, point_in_rect(0, 1, 0, 1), np.array([40, 0, 40])) 
        arr = self.icon.get_arr(img.shape[0])
        blend_icon(img, arr)
        
    def encode(self):
        return (OBJECT_TO_IDX["floor"], COLOR_TO_IDX[self.color], 0)

class HealthNode(WorldObj):
    def __init__(self):
        super().__init__("floor", "green")
        self.type = "health"
        # Custom path can be set here:
        self.icon = CachedIcon(pt.OutlineIcon.HEART_PLUS, "#00FF00", custom_path=None)
        
    def can_overlap(self):
        return True
        
    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), np.array([0, 40, 0])) 
        arr = self.icon.get_arr(img.shape[0])
        blend_icon(img, arr)
        
    def encode(self):
        return (OBJECT_TO_IDX["floor"], COLOR_TO_IDX[self.color], 0)

def draw_number_on_img(img_arr, text):
    """Draws a small sequence number on the tile image."""
    from PIL import ImageDraw, ImageFont
    h, w = img_arr.shape[:2]
    pil_img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(h * 0.45)))
    except:
        font = ImageFont.load_default()
    
    # Draw bottom right
    x, y = int(w * 0.6), int(h * 0.5)
    # Outline for visibility
    draw.text((x-1, y-1), text, font=font, fill=(0,0,0))
    draw.text((x+1, y-1), font=font, text=text, fill=(0,0,0))
    draw.text((x-1, y+1), font=font, text=text, fill=(0,0,0))
    draw.text((x+1, y+1), font=font, text=text, fill=(0,0,0))
    draw.text((x, y), text, font=font, fill=(255,255,255))
    img_arr[:, :, :] = np.array(pil_img)

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

class Destination(WorldObj):
    def __init__(self, color="blue", seq_id=None):
        super().__init__("box", color)
        self.type = "destination"
        self.seq_id = seq_id
        # Custom path can be set here:
        self.icon = CachedIcon(pt.OutlineIcon.TARGET, "#0000FF", custom_path=None)
        
    def can_overlap(self):
        return True
        
    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), np.array([0, 0, 40])) 
        arr = self.icon.get_arr(img.shape[0])
        blend_icon(img, arr)
        if self.seq_id is not None:
            draw_number_on_img(img, str(self.seq_id))
        
    def encode(self):
        return (OBJECT_TO_IDX["box"], COLOR_TO_IDX[self.color], 0)

class LevelGoal(Goal):
    def __init__(self):
        super().__init__()
        self.type = 'goal'
        # Custom path can be set here:
        self.icon = CachedIcon(pt.OutlineIcon.FLAG, "#00FF00", custom_path=None)

    def render(self, img):
        # We replace the default goal shape with a flag icon
        arr = self.icon.get_arr(img.shape[0])
        blend_icon(img, arr)
