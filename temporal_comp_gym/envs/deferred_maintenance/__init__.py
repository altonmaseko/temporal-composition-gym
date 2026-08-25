from .core import DeferredMaintenanceEnv as _CoreEnv
from .wrappers import DeferredMaintenanceObservationWrapper

class DeferredMaintenanceEnv(DeferredMaintenanceObservationWrapper):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, **kwargs):
        env = _CoreEnv(**kwargs)
        super().__init__(env)
