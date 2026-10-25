import pymunk

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

class PathBoundary:
    def __init__(self, space, rect):
        x, y, w, h = rect
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (x + w/2, y + h/2)
        self.shape = pymunk.Poly.create_box(self.body, (w, h))
        self.shape.sensor = True
        self.shape.collision_type = 2 # Path boundary / Safe Zone
        space.add(self.body, self.shape)
        self.rect = rect

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

class StageEntry:
    def __init__(self, space, rect):
        x, y, w, h = rect
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (x + w/2, y + h/2)
        self.shape = pymunk.Poly.create_box(self.body, (w, h))
        self.shape.sensor = True
        self.shape.collision_type = 6 # Entry
        space.add(self.body, self.shape)
        self.rect = rect

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
