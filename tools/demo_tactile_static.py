"""Static SO100 + tactile pad visualisation (3D taxel spheres + 2D heatmap panel).

Design notes:
    - Reuses the wrist_roll_tactile_pad_link from the URDF as the tactile anchor.
    - Uses a manual flat patch by default for easy offset tuning to align with target area.
    - Auto-follow contact box is disabled by default; just check whether taxel points sit flush.

Visualisations:
  1. IsaacSim 3D viewport: coloured taxel spheres (red=contact, blue=no contact) + XYZ axis markers.
  2. Docked omni.ui window: 12x32 viridis heatmap, updated every frame.

Usage:
    python tools/demo_tactile_static.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO100 static tactile pad visualisation")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--max-steps", type=int, default=0)
parser.add_argument("--vis-scale", type=int, default=12, help="Pixel upscale factor per taxel")
parser.add_argument("--auto-contact-demo", action="store_true", help="Enable legacy auto-follow contact box to generate artificial contact signal")
parser.add_argument("--check-host-penetration", action="store_true", help="Use Wrist_Roll_08c host mesh to check whether taxel penetration produces fn > 0")
parser.add_argument("--pad-offset-x-mm", type=float, default=0.0, help="Translate patch along pad local X axis (mm)")
parser.add_argument("--pad-offset-y-mm", type=float, default=0.0, help="Translate patch along pad local Y axis (mm)")
parser.add_argument("--pad-offset-z-mm", type=float, default=0.0, help="Translate patch along pad local Z axis (mm)")
parser.add_argument("--pad-rot-z-deg", type=float, default=0.0, help="Rotate patch around pad local Z axis (deg)")
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = False   # omni.ui heatmap panel requires GUI

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after AppLauncher ─────────────────────────────────────────────
import numpy as np


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from lerobot_robot_isaacsim_tactile.tactile_pad import (
    DEFAULT_TACTILE_COUNTS_PER_M,
    DEFAULT_TACTILE_DEBUG_POINT_RADIUS_M,
    DEFAULT_TACTILE_MAX_COUNTS,
    DEFAULT_TACTILE_SHELL_M,
    DEFAULT_TACTILE_IDLE_BOX_POS_W,
    DEFAULT_TACTILE_NORMAL_AXIS,
    DEFAULT_TACTILE_NORMAL_OFFSET_M,
    DEFAULT_TACTILE_PAD_LINK_NAME,
    DEFAULT_TACTILE_PATCH_OFFSET_POS_B,
    DEFAULT_TACTILE_PATCH_OFFSET_QUAT_B,
    DEFAULT_TACTILE_POINT_DISTANCE_M,
    DEFAULT_TACTILE_TARGET_LENGTH_M,
    DEFAULT_TACTILE_TARGET_WIDTH_M,
    resolve_tactile_target_mesh_prim_path,
)
from lerobot_robot_isaacsim_tactile.warp_sdf_tactile import WarpSdfTactileSensor, WarpSdfTactileSensorCfg

from lerobot_robot_isaacsim_tactile.scene import _ASSETS
_USD  = _ASSETS / "gripper_so100_tactile/usd/so100_follower_tactile.usd"

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# ── Sensor geometry (must match pad_on_wrist_roll.py) ────────────────────
_PAD_LINK = DEFAULT_TACTILE_PAD_LINK_NAME
# Auto-follow contact box: slightly smaller than taxel grid → edge taxels get fn=0, interior fn>0
_BOX_HALF = (0.004, 0.012, 0.004)


def _quat_mul_wxyz(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


# ── omni.ui heatmap panel ─────────────────────────────────────────────────
def _create_heatmap_panel(num_rows: int, num_cols: int, scale: int = 12):
    """Create a 12x32 viridis heatmap window docked to the right in IsaacSim."""
    import asyncio
    try:
        import omni.ui as ui
        import omni.kit.app
    except ImportError:
        print("[heatmap] omni.ui not available, skipping panel.")
        return None

    try:
        from matplotlib import cm
        cmap = cm.get_cmap("viridis")
    except ImportError:
        cmap = None

    h, w = num_rows * scale, num_cols * scale
    provider = ui.ByteImageProvider()
    blank = np.zeros((h, w, 4), dtype=np.uint8); blank[..., 3] = 255
    provider.set_bytes_data(blank.flatten().data, [w, h])

    window = ui.Window(
        "SO100 Tactile Heatmap (fn)",
        width=w + 60, height=h + 80, visible=True,
        dock_preference=ui.DockPreference.RIGHT_TOP,
    )
    with window.frame:
        with ui.VStack(spacing=4):
            ui.Label("Wrist_Roll_08c inner surface   12x32 taxels   viridis (0→1)")
            ui.ImageWithProvider(provider, width=w, height=h)

    async def _dock(title: str):
        app = omni.kit.app.get_app()
        for _ in range(60):
            if ui.Workspace.get_window(title): break
            await app.next_update_async()
        win = ui.Workspace.get_window(title)
        prop = ui.Workspace.get_window("Property")
        if win:
            if prop: win.dock_in(prop, ui.DockPosition.SAME, 1.0)
            win.visible = True; win.focus()

    asyncio.ensure_future(_dock(window.title))
    print(f"[heatmap] Created window '{window.title}'  ({w}x{h} px)")

    def _to_rgba(img01):
        img01 = np.clip(np.nan_to_num(img01), 0.0, 1.0)
        if cmap is None:
            g = (img01 * 255).astype(np.uint8)
            return np.dstack((g, g, g, np.full_like(g, 255)))
        rgb = (cmap(img01)[..., :3] * 255).astype(np.uint8)
        return np.concatenate((rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)), axis=2)

    def update(fn_grid: np.ndarray):
        up = np.kron(fn_grid.astype(np.float32), np.ones((scale, scale), np.float32))
        provider.set_bytes_data(_to_rgba(up).flatten().data, [w, h])

    return {"window": window, "provider": provider, "update": update}


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    sim_dt = 1.0 / 60.0
    sim = SimulationContext(SimulationCfg(dt=sim_dt, device=args_cli.device))

    # ── World base prims ─────────────────────────────────────────────────
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    tx, ty, tz = 0.8, 0.6, 0.72
    sim_utils.CuboidCfg(
        size=(tx, ty, tz),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
    ).func("/World/Table", sim_utils.CuboidCfg(size=(tx, ty, tz)),
           translation=(0.0, 0.0, tz * 0.5))

    # ── Robot arm ─────────────────────────────────────────────────────────
    arm = Articulation(cfg=ArticulationCfg(
        prim_path="/World/Arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_USD),
            activate_contact_sensors=True,   # required for ContactSensor pose tracking
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, fix_root_link=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, tz), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={j: 0.0 for j in _JOINTS},
        ),
        actuators={"all": ImplicitActuatorCfg(
            joint_names_expr=list(_JOINTS),
            stiffness=200.0, damping=10.0, effort_limit=10.0)},
    ))

    pad_offset_b = (
        DEFAULT_TACTILE_PATCH_OFFSET_POS_B[0] + args_cli.pad_offset_x_mm * 1.0e-3,
        DEFAULT_TACTILE_PATCH_OFFSET_POS_B[1] + args_cli.pad_offset_y_mm * 1.0e-3,
        DEFAULT_TACTILE_PATCH_OFFSET_POS_B[2] + args_cli.pad_offset_z_mm * 1.0e-3,
    )
    pad_rot_z_rad = math.radians(float(args_cli.pad_rot_z_deg))
    half_yaw = 0.5 * pad_rot_z_rad
    pad_quat_delta_b = (
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )
    pad_quat_b = _quat_mul_wxyz(DEFAULT_TACTILE_PATCH_OFFSET_QUAT_B, pad_quat_delta_b)
    row_pitch_m = DEFAULT_TACTILE_TARGET_WIDTH_M / 12.0
    col_pitch_m = DEFAULT_TACTILE_TARGET_LENGTH_M / 32.0
    target_mesh_prim_path = None
    if args_cli.check_host_penetration:
        target_mesh_prim_path = resolve_tactile_target_mesh_prim_path(arm_prim_path="/World/Arm")

    # ── Tactile sensor (manual flat patch; width derived from 2.8 cm target) ──
    tactile = WarpSdfTactileSensor(cfg=WarpSdfTactileSensorCfg(
        prim_path="/World/Arm",
        elastomer_prim_paths=[f"/World/Arm/{_PAD_LINK}"],
        target_mesh_prim_path=target_mesh_prim_path,
        # Same contact model as the robot (tactile_pad.py), shown as counts / 255.
        mesh_use_signed_distance=False,
        mesh_signed_distance_method="normal",
        mesh_shell_thickness=DEFAULT_TACTILE_SHELL_M,
        num_rows=12,
        num_cols=32,
        point_distance=min(DEFAULT_TACTILE_POINT_DISTANCE_M, col_pitch_m),
        normal_axis=DEFAULT_TACTILE_NORMAL_AXIS,
        normal_offset=DEFAULT_TACTILE_NORMAL_OFFSET_M,
        patch_offset_pos_b=pad_offset_b,
        patch_offset_quat_b=pad_quat_b,
        # No contact by default; legacy box demo only enabled with --auto-contact-demo.
        box_pos_w=DEFAULT_TACTILE_IDLE_BOX_POS_W,
        box_quat_w=(1.0, 0.0, 0.0, 0.0),
        box_half_extents=_BOX_HALF,
        stiffness=DEFAULT_TACTILE_COUNTS_PER_M,
        max_force=DEFAULT_TACTILE_MAX_COUNTS,
        normalize_forces=True,
        # 3D taxel sphere markers (axes off to avoid occluding contact view)
        debug_vis=True,
        debug_vis_show_all_taxels=True,     # blue=no contact, red=contact
        debug_vis_show_axes=False,
        debug_vis_axes_scale=0.025,
        debug_vis_point_radius=DEFAULT_TACTILE_DEBUG_POINT_RADIUS_M,
    ))

    # ── reset ─────────────────────────────────────────────────────────────
    sim.reset()
    arm.reset()
    tactile.reset()
    sim.set_camera_view(eye=(0.20, -0.30, 1.0), target=(0.02, 0.0, 0.85))

    # ── Body indices ──────────────────────────────────────────────────────
    joint_names = list(arm.data.joint_names)
    joint_idx   = {n: i for i, n in enumerate(joint_names)}
    body_names  = list(arm.data.body_names)
    print(f"\n[demo] joints: {joint_names}")
    print(f"[demo] bodies: {body_names}")

    if _PAD_LINK in body_names:
        pad_idx = body_names.index(_PAD_LINK)
        print(f"[demo] {_PAD_LINK} body_idx={pad_idx} ✓")
    else:
        pad_idx = None
        print(f"[demo] WARNING: {_PAD_LINK} not found in body_names!")

    if target_mesh_prim_path is not None:
        print(f"[demo] host-penetration check mesh: {target_mesh_prim_path}")

    print(
        "[demo] manual patch: width=28.0 mm across 12 rows, "
        "32 cols tiled with the same sphere diameter, "
        f"row_pitch={DEFAULT_TACTILE_POINT_DISTANCE_M * 1.0e3:.3f} mm, "
        f"col_pitch={col_pitch_m * 1.0e3:.3f} mm, "
        f"sphere_radius={DEFAULT_TACTILE_DEBUG_POINT_RADIUS_M * 1.0e3:.3f} mm, "
        f"length={DEFAULT_TACTILE_TARGET_LENGTH_M * 1.0e3:.3f} mm, "
        f"default_pad_quat={DEFAULT_TACTILE_PATCH_OFFSET_QUAT_B}, "
        f"pad_rot_z_deg={args_cli.pad_rot_z_deg:+.1f}, "
        f"pad_offset_b=({pad_offset_b[0]:+.4f}, {pad_offset_b[1]:+.4f}, {pad_offset_b[2]:+.4f}) m"
    )

    # ── 2D heatmap panel ──────────────────────────────────────────────────
    panel = _create_heatmap_panel(num_rows=12, num_cols=32, scale=args_cli.vis_scale)
    if panel is None:
        print("[demo] No heatmap panel, relying on 3D taxel sphere visualisation only.")

    target = arm.data.joint_pos.clone()   # hold at zero pose
    step   = 0
    max_steps = args_cli.max_steps

    print("[demo] Static display — rotate in viewport to inspect taxel sphere positions.")
    if args_cli.auto_contact_demo:
        print("[demo] auto-contact-demo enabled: contact box tracks pad, red dots = artificial contact.\n")
    elif args_cli.check_host_penetration:
        print("[demo] host-penetration check enabled: taxels entering Wrist_Roll_08c mesh should produce fn>0.\n")
    else:
        print("[demo] Default mode: no auto-follow box, taxels should appear mostly blue.\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    while simulation_app.is_running() and (max_steps == 0 or step < max_steps):
        # Hold zero pose
        arm.set_joint_position_target(target)
        arm.write_data_to_sim()
        sim.step()
        arm.update(sim_dt)

        # auto-follow: only track pad link world pose when explicitly requested
        if args_cli.auto_contact_demo and pad_idx is not None:
            # body_state_w: [E, B, 13]  pos=[:3]  quat(w,x,y,z)=[3:7]
            pad_pos_w = arm.data.body_state_w[0, pad_idx, :3]
            pad_quat_w = arm.data.body_state_w[0, pad_idx, 3:7]
            tactile.set_box_pose(pad_pos_w, pad_quat_w)

        # Update sensor
        tactile.update(dt=sim_dt, force_recompute=True)

        # Read fn values  (E=1, S=1, P=384, 4)
        tp = tactile.data.tactile_points_w_per_sensor
        if tp is not None:
            fn_grid = tp[0, 0, :, 3].detach().float().clamp_min(0.0).cpu().numpy().reshape(12, 32)
        else:
            fn_grid = np.zeros((12, 32), dtype=np.float32)

        if panel is not None:
            panel["update"](fn_grid)

        if step % 60 == 0:
            nz = int((fn_grid > 1e-6).sum())
            print(f"[demo] step={step:5d}  fn max={fn_grid.max():.4f}  "
                  f"mean={fn_grid.mean():.4f}  nonzero={nz}/384")

        step += 1

    print(f"\n[demo] Done — {step} frames.")


if __name__ == "__main__":
    main()
    simulation_app.close()
