"""Gym wrapper that runs a reward machine alongside an environment.

obs_mode
  None      observation unchanged (QRM reads the RM state from info["rm"])
  "append"  concatenate RM features to a vector observation (PPO-RM, DQN-TS delivery)
  "dict"    {"image": obs, "task_state": features} (DQN-TS colour: features join after the CNN)
reward_mode
  "env"     return the env reward (the RM reward is in info["rm"]["reward"])
  "rm"      return the RM reward (equal to the env reward on real trajectories; checked)
"""
import gymnasium as gym
import numpy as np

from .colour import CueRecordingRM
from .delivery import DeliveryRM
from .maintenance import MaintenanceRM

_RM_BY_ENV = {"delivery": DeliveryRM, "colour": CueRecordingRM, "maintenance": MaintenanceRM}


def make_reward_machine(env_key, env):
    return _RM_BY_ENV[env_key].from_env(env)


class RewardMachineWrapper(gym.Wrapper):
    def __init__(self, env, rm, obs_mode=None, reward_mode="env", check_reward=True, atol=1e-6):
        super().__init__(env)
        if obs_mode not in (None, "append", "dict"):
            raise ValueError("obs_mode must be None, 'append' or 'dict'")
        if reward_mode not in ("env", "rm"):
            raise ValueError("reward_mode must be 'env' or 'rm'")
        self.rm = rm
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.check_reward = check_reward
        self.atol = atol
        self.state = None
        base = env.observation_space
        feat = gym.spaces.Box(0.0, 1.0, shape=(rm.feature_size,), dtype=np.float32)
        if obs_mode == "append":
            if len(base.shape) != 1:
                raise ValueError("obs_mode='append' needs a vector observation")
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(base.shape[0] + rm.feature_size,),
                                                    dtype=np.float32)
        elif obs_mode == "dict":
            self.observation_space = gym.spaces.Dict({"image": base, "task_state": feat})

    def _obs(self, obs):
        if self.obs_mode is None:
            return obs
        f = self.rm.features(self.state)
        if self.obs_mode == "append":
            return np.concatenate([np.asarray(obs, dtype=np.float32), f]).astype(np.float32)
        return {"image": obs, "task_state": f}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.state = self.rm.initial_state(info.get("events", []))
        info = dict(info)
        info["rm"] = {"state": self.state, "head": self.rm.head(self.state)}
        return self._obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        prev = self.state
        nxt, rm_reward, rm_terminal = self.rm.step(prev, info["events"])
        if self.check_reward and abs(rm_reward - reward) > self.atol:
            raise RuntimeError(f"RM reward {rm_reward} != env reward {reward} for events {info['events']}")
        if rm_terminal != terminated:
            raise RuntimeError(f"RM terminal {rm_terminal} != env terminated {terminated} for events {info['events']}")
        self.state = nxt
        info = dict(info)
        info["raw_reward"] = reward
        info["rm"] = {
            "state": prev, "next_state": nxt, "head": self.rm.head(prev), "next_head": self.rm.head(nxt),
            "reward": rm_reward, "terminal": rm_terminal,
        }
        if hasattr(self.rm, "transition_extras"):
            info["rm"]["extras"] = self.rm.transition_extras(prev, info["events"], nxt)
        out_reward = rm_reward if self.reward_mode == "rm" else reward
        return self._obs(obs), out_reward, terminated, truncated, info
