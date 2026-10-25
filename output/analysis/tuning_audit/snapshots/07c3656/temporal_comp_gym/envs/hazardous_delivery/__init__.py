from .core import HazardousDeliveryCoreEnv
from .wrappers import HazardousDeliveryObservationWrapper

def HazardousDeliveryEnv(**kwargs):
    include_health = kwargs.pop('include_health_in_state', True)
    include_inventory = kwargs.pop('include_inventory_in_state', True)
    
    env = HazardousDeliveryCoreEnv(**kwargs)
    
    env = HazardousDeliveryObservationWrapper(
        env,
        include_health_in_state=include_health,
        include_inventory_in_state=include_inventory
    )
    
    return env
