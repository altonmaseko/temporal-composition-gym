"""Reward machines for the three tasks (protocol v2 section 4).

Each machine is driven only by the environment's info["events"] stream and its fixed
configuration. The env reward and the RM reward are equal on every real step: the RM adds
task state (and, for QRM, counterfactual updates), never reward shaping.
"""
from .colour import CueRecordingRM
from .delivery import DeliveryRM
from .maintenance import MaintenanceRM
from .wrappers import RewardMachineWrapper, make_reward_machine

__all__ = ["CueRecordingRM", "DeliveryRM", "MaintenanceRM", "RewardMachineWrapper", "make_reward_machine"]
