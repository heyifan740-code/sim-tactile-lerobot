"""Re-generate the arm USD shipped in the plugin from its URDF (IsaacLab UrdfConverter).

Usage:
    python tools/convert_urdf_to_usd.py [--out-dir DIR] [--collider-type convex_decomposition|convex_hull]

The defaults reproduce the shipped asset in src/lerobot_robot_isaacsim_tactile/assets/gripper_so100_tactile/usd/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

ASSET = Path(__file__).resolve().parents[1] / "src/lerobot_robot_isaacsim_tactile/assets/gripper_so100_tactile"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out-dir", type=str, default=str(ASSET / "usd"))
parser.add_argument("--collider-type", choices=["convex_decomposition", "convex_hull"], default="convex_decomposition")
args = parser.parse_args()
app = AppLauncher(headless=True).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

cfg = UrdfConverterCfg(
    asset_path=str(ASSET / "urdf/so100_follower_tactile.urdf"),
    usd_dir=args.out_dir,
    usd_file_name="so100_follower_tactile.usd",
    force_usd_conversion=True,
    make_instanceable=True,
    fix_base=False,
    merge_fixed_joints=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
    ),
    # The fixed jaw (wrist_roll_08c) is concave: its convex hull bulges up to ~24 mm over the
    # tactile pad, so objects pressed onto the pad stop in mid-air. Decomposition follows the pad face.
    collider_type=args.collider_type,
    self_collision=False,
)
print(f"[convert] {UrdfConverter(cfg).usd_path}  (collider_type={args.collider_type})", flush=True)
app.close()
