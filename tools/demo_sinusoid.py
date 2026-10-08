"""Minimal test — drive SO100 joints with sinusoids in IsaacSim.

This demo does NOT depend on lerobot. It calls build_scene() directly
so you can verify the USD loads, joints move, and the camera is live
before wiring up the full lerobot follower class.

Usage:
    python tools/demo_sinusoid.py
    python tools/demo_sinusoid.py --headless --max-steps 120
    (--enable_cameras is forced on automatically since the camera is always used)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO100 sim sinusoid test (no lerobot required).")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--max-steps", type=int, default=0)
parser.add_argument("--freq", type=float, default=0.3)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # camera is always used in this demo
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- post-AppLauncher imports ---
import torch
import numpy as np


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext, SimulationCfg

from lerobot_robot_isaacsim_tactile.scene import _ASSETS
_USD = _ASSETS / "gripper_so100_tactile/usd/so100_follower_tactile.usd"

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_LIMITS = {
    "shoulder_pan":  (-1.91,  1.91),
    "shoulder_lift": (-1.74,  1.74),
    "elbow_flex":    (-1.69,  1.69),
    "wrist_flex":    (-1.65,  1.65),
    "wrist_roll":    (-2.74,  2.84),
    "gripper":       (-0.17,  1.74),
}


def _sinusoid(joint: str, t: float, freq: float) -> float:
    lo, hi = _LIMITS[joint]
    mid = (lo + hi) / 2.0
    amp = (hi - lo) / 2.0 * 0.5
    return mid + amp * math.sin(2.0 * math.pi * freq * t)


def main() -> None:
    sim_dt = 1.0 / 60.0
    sim = SimulationContext(SimulationCfg(dt=sim_dt, device=args_cli.device))

    # ground + light
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0)
    )

    # table
    tx, ty, tz = 0.8, 0.6, 0.72
    sim_utils.CuboidCfg(
        size=(tx, ty, tz),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
    ).func("/World/Table", sim_utils.CuboidCfg(size=(tx, ty, tz)), translation=(0.0, 0.0, tz * 0.5))

    # arm
    arm_cfg = ArticulationCfg(
        prim_path="/World/Arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, tz),  # sit on table top
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={j: 0.0 for j in _JOINTS},
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=list(_JOINTS),
                stiffness=200.0,
                damping=10.0,
                effort_limit=10.0,
            )
        },
    )
    arm = Articulation(cfg=arm_cfg)

    # top camera
    cam_cfg = CameraCfg(
        prim_path="/World/TopCamera",
        update_period=0,
        data_types=["rgb"],
        width=640,
        height=480,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=0.8,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.8, 1.3),
            rot=(0.7071, 0.0, 0.0, -0.7071),
            convention="world",
        ),
    )
    camera = Camera(cfg=cam_cfg)

    sim.reset()
    arm.reset()
    camera.reset()
    sim.set_camera_view(eye=(0.0, -1.2, 1.0), target=(0.0, 0.0, 0.7))

    joint_names = list(arm.data.joint_names)
    joint_idx = {n: i for i, n in enumerate(joint_names)}
    print(f"\n[demo] Joint names in USD: {joint_names}")
    missing = [j for j in _JOINTS if j not in joint_idx]
    if missing:
        print(f"[demo] WARNING: missing joints: {missing}")
    else:
        print("[demo] All 6 SO100 joints found ✓")

    target = arm.data.joint_pos.clone()
    step, t = 0, 0.0
    max_steps = args_cli.max_steps

    while simulation_app.is_running() and (max_steps == 0 or step < max_steps):
        for j in _JOINTS:
            if j in joint_idx:
                target[0, joint_idx[j]] = _sinusoid(j, t, args_cli.freq)

        arm.set_joint_position_target(target)
        arm.write_data_to_sim()
        sim.step()
        arm.update(sim_dt)
        camera.update(dt=sim_dt)

        if step % 60 == 0:
            pos = arm.data.joint_pos[0]
            vals = {j: f"{pos[joint_idx[j]].item():.3f}" for j in _JOINTS if j in joint_idx}
            print(f"[demo] step={step:4d}  joints={vals}")
            rgb = camera.data.output.get("rgb")
            if rgb is not None:
                print(f"         camera rgb: shape={tuple(rgb.shape)}  "
                      f"mean={rgb.float().mean():.1f}  max={rgb.float().max():.1f}")

        step += 1
        t += sim_dt

    print(f"\n[demo] done — {step} steps. USD loaded OK.")


if __name__ == "__main__":
    main()
    simulation_app.close()
