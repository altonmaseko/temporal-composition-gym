from gymnasium.envs.registration import register

register(
    id='TemporalComp/HazardousDelivery-v0',
    entry_point='temporal_comp_gym.envs:HazardousDeliveryEnv',
)

register(
    id='TemporalComp/DeferredMaintenance-v0',
    entry_point='temporal_comp_gym.envs:DeferredMaintenanceEnv',
)

register(
    id='TemporalComp/SequentialColour-v0',
    entry_point='temporal_comp_gym.envs:SequentialColourEnv',
)
