"""LeRobot robot plugin: Isaac Sim SO100 follower with a FlexiTac tactile pad.

Installing this package makes `--robot.type=isaacsim_so100_tactile_follower` available
to the official lerobot scripts (`lerobot-teleoperate`, `lerobot-record`, ...), which
discover `lerobot_robot_*` packages via `register_third_party_plugins()`.
Isaac Sim / IsaacLab are imported only inside `connect()`.
"""

from .config_isaacsim_so100_tactile_follower import IsaacSimSO100TactileFollowerConfig, SimTactilePadConfig
from .isaacsim_so100_tactile_follower import IsaacSimSO100TactileFollower

__all__ = ["IsaacSimSO100TactileFollower", "IsaacSimSO100TactileFollowerConfig", "SimTactilePadConfig"]
