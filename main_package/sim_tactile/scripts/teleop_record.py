"""Real SO100 leader  →  IsaacSim SO100-tactile follower, recording via lerobot.

Thin wrapper around the external lerobot_dev record entrypoint. The only
tricks are:

1. Put `/home/fan/workspace/lerobot_dev/src` on `sys.path` (or a path
    provided via `LEROBOT_DEV_ROOT` / `LEROBOT_DEV_SRC`).
2. Import `main_package.sim_tactile.follower` before lerobot parses the
    CLI, so `--robot.type=isaacsim_so100_tactile_follower` resolves.

Usage (example — fill in paths / ports):

    ./isaaclab.sh -p main_package/sim_tactile/scripts/teleop_record.py \\
        --robot.type=isaacsim_so100_tactile_follower \\
        --robot.urdf_path=main_package/sim_tactile/assets/gripper_so100_tactile/urdf/so100_follower_tactile.urdf \\
        --teleop.type=so100_leader \\
        --teleop.port=/dev/ttyACM0 \\
        --display_data=true \\
        --dataset.repo_id=local/sim_so100_tactile_demo \\
        --dataset.num_episodes=5 \\
        --dataset.single_task="peg insertion"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_lerobot_dev_src() -> Path:
    env_src = os.environ.get("LEROBOT_DEV_SRC")
    if env_src:
        path = Path(env_src).expanduser().resolve()
        if path.is_dir():
            return path

    env_root = os.environ.get("LEROBOT_DEV_ROOT")
    if env_root:
        path = (Path(env_root).expanduser().resolve() / "src")
        if path.is_dir():
            return path

    workspace_default = Path(__file__).resolve().parents[4] / "lerobot_dev" / "src"
    if workspace_default.is_dir():
        return workspace_default

    hardcoded_default = Path("/home/fan/workspace/lerobot_dev/src")
    if hardcoded_default.is_dir():
        return hardcoded_default.resolve()

    raise FileNotFoundError(
        "Could not locate lerobot_dev/src. Set LEROBOT_DEV_ROOT or LEROBOT_DEV_SRC to your lerobot_dev checkout."
    )


def _bootstrap_lerobot_dev() -> None:
    src_path = _resolve_lerobot_dev_src()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _main() -> None:
    _bootstrap_lerobot_dev()

    # Importing the follower package has the side-effect of registering the
    # IsaacSim follower subclass with RobotConfig's draccus registry.
    import main_package.sim_tactile.follower  # noqa: F401

    from lerobot.scripts.lerobot_record import main as lerobot_record_main

    lerobot_record_main()


if __name__ == "__main__":
    _main()
