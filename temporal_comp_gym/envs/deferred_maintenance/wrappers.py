import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DeferredMaintenanceObservationWrapper(gym.ObservationWrapper):
    """Flattens the core env's dict observation into the 1D vector described in
    agent_observation_space.txt (raw units: pixels, pixels/s, radians)."""

    def __init__(self, env):
        super().__init__(env)

        # Determine the size of the micro-navigation sensors
        spatial_sensor_type = self.unwrapped.spatial_sensor_type
        if spatial_sensor_type == 'local_grid':
            micro_nav_size = 9 # 3x3 local grid of cell types
        else:
            micro_nav_size = 8 # 8 simulated ray-cast lines

        # Determine the size of station radar (X and Y per color)
        self.num_colors = self.unwrapped.max_code_types
        radar_size = self.num_colors * 2

        # Kinematics(6) + MicroNav(9/8) + MacroNav(2) + FloorCode(1) + Readiness(1) + Radar
        obs_size = 6 + micro_nav_size + 2 + 1 + 1 + radar_size
        self.micro_nav_size = micro_nav_size

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

    def observation(self, obs):
        kinematics = np.array([
            obs['pos_x'], obs['pos_y'],
            obs['vel_x'], obs['vel_y'],
            obs['angle'], obs['ang_vel']
        ], dtype=np.float32)

        micro_nav = np.array(obs['micro_nav'], dtype=np.float32).flatten()
        macro_nav = np.array(obs['macro_nav'], dtype=np.float32).flatten()

        floor_code = np.array([obs['floor_code']], dtype=np.float32)
        readiness = np.array([obs['readiness']], dtype=np.float32)

        station_radar = np.array(obs['station_radar'], dtype=np.float32).flatten()

        flat_obs = np.concatenate([
            kinematics,
            micro_nav,
            macro_nav,
            floor_code,
            readiness,
            station_radar
        ])

        return flat_obs


class MaintenanceScaledObservation(gym.ObservationWrapper):
    """Fixed, deterministic rescaling of the flat maintenance observation (protocol v2).

    Applied identically for PPO, RecurrentPPO and PPO-RM, so no running statistics are
    learned or need saving. Positions and offsets are expressed in cells, velocities in
    cells/s / 4, angular velocity / 10, cell-type codes / 7, floor code / K. The radar's
    -99 "absent" sentinel is replaced by a presence flag plus a zero offset, giving three
    values per station type instead of two.
    """

    def __init__(self, env):
        super().__init__(env)
        core = env.unwrapped
        self.cell = float(core.cell_size)
        self.k = int(core.max_code_types)
        self.micro = 9 if core.spatial_sensor_type == "local_grid" else 8
        self.grid_sensor = core.spatial_sensor_type == "local_grid"
        size = 6 + self.micro + 2 + 1 + 1 + 3 * self.k
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)

    def observation(self, obs):
        c, k, m = self.cell, self.k, self.micro
        kin = obs[:6].astype(np.float64)
        kin = np.array([kin[0] / c, kin[1] / c, kin[2] / (4 * c), kin[3] / (4 * c), kin[4], kin[5] / 10.0])
        micro = obs[6:6 + m] / 7.0 if self.grid_sensor else obs[6:6 + m]
        macro = obs[6 + m:8 + m] / c
        floor = obs[8 + m:9 + m] / k
        ready = obs[9 + m:10 + m]
        radar_raw = obs[10 + m:10 + m + 2 * k].reshape(k, 2)
        radar = []
        for dx, dy in radar_raw:
            if dx == -99.0 and dy == -99.0:
                radar.extend([0.0, 0.0, 0.0])
            else:
                radar.extend([1.0, dx / c, dy / c])
        return np.concatenate([kin, micro, macro, floor, ready, radar]).astype(np.float32)
