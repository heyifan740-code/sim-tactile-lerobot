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
JAW_LINKS = ("gripper_link", "moving_jaw_link")

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
    # The fixed jaw is concave; its convex hull would cover the tactile pad.
    collider_type=args.collider_type,
    self_collision=False,
)
usd_path = UrdfConverter(cfg).usd_path

# Tighten the jaw colliders: default convex decomposition sits ~1.5 mm proud of the jaw faces, so objects
# would stop above the tactile pad. shrinkWrap projects the hull vertices back onto the mesh surface.
if args.collider_type == "convex_decomposition":
    from pxr import PhysxSchema, Usd, UsdPhysics

    physics = Path(usd_path).parent / "configuration" / (Path(usd_path).stem + "_physics.usd")
    stage = Usd.Stage.Open(str(physics))
    for prim in stage.Traverse():
        if any(f"/colliders/{link}/" in str(prim.GetPath()) for link in JAW_LINKS) and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            api.CreateShrinkWrapAttr(True)
            api.CreateMaxConvexHullsAttr(64)
            api.CreateVoxelResolutionAttr(2_000_000)
            api.CreateErrorPercentageAttr(1.0)
            print(f"[convert] tight decomposition: {prim.GetPath()}", flush=True)
    stage.GetRootLayer().Save()
print(f"[convert] {usd_path}  (collider_type={args.collider_type})", flush=True)
app.close()
