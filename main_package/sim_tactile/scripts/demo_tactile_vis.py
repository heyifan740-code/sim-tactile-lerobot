"""SO100 sinusoid + wrist-roll tactile pad visualisation.

The WarpSdfTactileSensor is attached to ``wrist_roll_tactile_pad_link``.
An analytic box target auto-follows the pad link's world pose every step
(centred at the pad origin, oriented with the pad frame), so the sensor
always produces a spatially-varying force map — no need to manually move
objects into contact.

Box half-extents (0.004 m × 0.012 m × 0.004 m) are chosen to be smaller
than the taxel grid (11 mm × 31 mm at 1 mm pitch), creating visible edge
effects: inner taxels are in contact (fn > 0), outer taxels are not (fn = 0).

Outputs
-------
Console   – fn max / mean / nonzero count every ``--vis-interval`` steps.
PNG       – viridis heatmap saved to /tmp/tactile_vis/tactile_NNNNN.png
            every ``--vis-interval`` steps.
3-D       – coloured taxel spheres in the IsaacSim viewport when ``--debug-vis``
            is passed (GUI mode only).

Usage
-----
# GUI + 3-D taxel markers:
    ./isaaclab.sh -p main_package/sim_tactile/scripts/demo_tactile_vis.py --debug-vis

# Headless:
    ./isaaclab.sh -p main_package/sim_tactile/scripts/demo_tactile_vis.py \\
        --headless --max-steps 300
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO100 tactile pad visualisation demo.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--max-steps", type=int, default=0, help="0 = run until window closed.")
parser.add_argument("--freq", type=float, default=0.3, help="Sinusoid frequency (Hz).")
parser.add_argument("--debug-vis", action="store_true",
                    help="Enable 3-D taxel sphere markers (GUI mode only).")
parser.add_argument("--vis-interval", type=int, default=30,
                    help="Save PNG heatmap every N steps.")
args_cli = parser.parse_args()
args_cli.enable_cameras = True   # camera sensor always requires this

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-AppLauncher imports ────────────────────────────────────────────────
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.warp_sdf_tactile import WarpSdfTactileSensor, WarpSdfTactileSensorCfg
from isaaclab.sim import SimulationContext, SimulationCfg

_REPO = Path(__file__).resolve().parents[3]
_USD  = _REPO / "main_package/sim_tactile/assets/gripper_so100_tactile/usd/so100_follower_tactile.usd"

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_LIMITS = {
    "shoulder_pan":  (-1.91,  1.91),
    "shoulder_lift": (-1.74,  1.74),
    "elbow_flex":    (-1.69,  1.69),
    "wrist_flex":    (-1.65,  1.65),
    "wrist_roll":    (-2.74,  2.84),
    "gripper":       (-0.17,  1.74),
}

# Box half-extents for the auto-follow analytic SDF target (metres).
# Smaller than the 11 mm × 31 mm taxel grid → edge taxels fall outside → fn=0
# there while inner taxels give fn > 0, producing visible spatial variation.
_BOX_HALF = (0.004, 0.012, 0.004)
_PAD_LINK  = "wrist_roll_tactile_pad_link"


def _sinusoid(joint: str, t: float, freq: float) -> float:
    lo, hi = _LIMITS[joint]
    mid = (lo + hi) / 2.0
    amp = (hi - lo) / 2.0 * 0.5
    return mid + amp * math.sin(2.0 * math.pi * freq * t)


def _save_heatmap(fn_grid: np.ndarray, step: int, out_dir: Path) -> Path:
    """Render fn_grid (12, 32) as a viridis PNG and return its path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = fn_grid.shape
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(fn_grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto",
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fn  (normalized 0–1)")
    ax.set_title(
        f"Wrist-roll tactile pad   step={step}"
        f"   max={fn_grid.max():.3f}   mean={fn_grid.mean():.4f}"
        f"   nonzero={int((fn_grid > 1e-6).sum())}/{rows * cols}"
    )
    ax.set_xlabel(f"cols  ({cols})")
    ax.set_ylabel(f"rows  ({rows})")
    out_path = out_dir / f"tactile_{step:05d}.png"
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    out_dir = Path("/tmp/tactile_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_dt = 1.0 / 60.0
    sim = SimulationContext(SimulationCfg(dt=sim_dt, device=args_cli.device))

    # ── world prims ──────────────────────────────────────────────────────────
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0)
    )
    tx, ty, tz = 0.8, 0.6, 0.72
    sim_utils.CuboidCfg(
        size=(tx, ty, tz),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
    ).func("/World/Table", sim_utils.CuboidCfg(size=(tx, ty, tz)),
           translation=(0.0, 0.0, tz * 0.5))

    # ── arm ──────────────────────────────────────────────────────────────────
    arm_cfg = ArticulationCfg(
        prim_path="/World/Arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_USD),
            # Required: adds PhysxContactReportAPI to all rigid bodies so the
            # WarpSdfTactileSensor's internal ContactSensor can track pad pose.
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, tz),
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

    # ── top camera ───────────────────────────────────────────────────────────
    camera = Camera(cfg=CameraCfg(
        prim_path="/World/TopCamera",
        update_period=0,
        data_types=["rgb"],
        width=640,
        height=480,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=0.8,
            horizontal_aperture=20.955, clipping_range=(0.01, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.8, 1.3),
            rot=(0.7071, 0.0, 0.0, -0.7071),
            convention="world",
        ),
    ))

    # ── tactile sensor ───────────────────────────────────────────────────────
    tactile = WarpSdfTactileSensor(cfg=WarpSdfTactileSensorCfg(
        prim_path="/World/Arm",
        elastomer_prim_paths=[f"/World/Arm/{_PAD_LINK}"],
        # 12 × 32 grid, 1 mm pitch — matches real driver.
        num_rows=12,
        num_cols=32,
        point_distance=0.001,
        # Local-Z is pad normal; taxel points are 2 mm from link origin along -Z.
        normal_axis=2,
        normal_offset=-0.002,
        patch_offset_pos_b=(0.0, 0.0, 0.0),
        patch_offset_quat_b=(1.0, 0.0, 0.0, 0.0),
        # Initial box (overwritten each step by auto-follow logic below).
        box_pos_w=(0.0, 0.0, tz),
        box_quat_w=(1.0, 0.0, 0.0, 0.0),
        box_half_extents=_BOX_HALF,
        # Spring model: 1 mm penetration → fn_norm ≈ 0.5 (no saturation).
        stiffness=500.0,
        max_force=1.0,
        normalize_forces=True,
        # 3-D taxel markers in IsaacSim viewport (GUI only).
        debug_vis=args_cli.debug_vis,
        debug_vis_show_all_taxels=True,  # show all taxels (blue = no contact)
    ))

    # ── reset ─────────────────────────────────────────────────────────────────
    sim.reset()
    arm.reset()
    camera.reset()
    tactile.reset()
    sim.set_camera_view(eye=(0.0, -1.2, 1.0), target=(0.0, 0.0, 0.7))

    # ── joint index map ───────────────────────────────────────────────────────
    joint_names = list(arm.data.joint_names)
    joint_idx   = {n: i for i, n in enumerate(joint_names)}
    print(f"\n[tactile_demo] Joints in USD: {joint_names}")
    missing = [j for j in _JOINTS if j not in joint_idx]
    if missing:
        print(f"[tactile_demo] WARNING missing joints: {missing}")
    else:
        print("[tactile_demo] All 6 SO100 joints found ✓")

    # ── body index for pad pose tracking ─────────────────────────────────────
    body_names = list(arm.data.body_names)
    if _PAD_LINK in body_names:
        pad_body_idx = body_names.index(_PAD_LINK)
        print(f"[tactile_demo] Pad body '{_PAD_LINK}' at body_idx={pad_body_idx} ✓")
    else:
        pad_body_idx = None
        print(f"[tactile_demo] WARNING: '{_PAD_LINK}' NOT found in body_names.")
        print(f"[tactile_demo]   body_names = {body_names}")
        print("[tactile_demo]   Box will stay at table-top — signal may be zero.")

    target   = arm.data.joint_pos.clone()
    step, t  = 0, 0.0
    max_steps = args_cli.max_steps
    print(f"[tactile_demo] PNG heatmaps → {out_dir}\n")

    # ── main loop ─────────────────────────────────────────────────────────────
    while simulation_app.is_running() and (max_steps == 0 or step < max_steps):
        # --- drive joints with sinusoids ---
        for j in _JOINTS:
            if j in joint_idx:
                target[0, joint_idx[j]] = _sinusoid(j, t, args_cli.freq)
        arm.set_joint_position_target(target)
        arm.write_data_to_sim()
        sim.step()
        arm.update(sim_dt)
        camera.update(dt=sim_dt)

        # --- auto-follow: box tracks pad link pose in world frame ---
        if pad_body_idx is not None:
            # body_state_w: [E, B, 13]  pos=[:3]  quat(w,x,y,z)=[3:7]
            pad_pos  = arm.data.body_state_w[0, pad_body_idx, :3]   # (3,)
            pad_quat = arm.data.body_state_w[0, pad_body_idx, 3:7]  # (w,x,y,z)
            tactile.set_box_pose(pad_pos, pad_quat)

        # --- update sensor ---
        tactile.update(dt=sim_dt, force_recompute=True)

        # --- read fn grid ---
        tp = tactile.data.tactile_points_w_per_sensor   # (E, S, P, 4) or None
        if tp is not None:
            fn = tp[0, 0, :, 3].detach().float().clamp_min(0.0).cpu().numpy()
            fn_grid = fn.reshape(12, 32)
        else:
            fn_grid = np.zeros((12, 32), dtype=np.float32)

        # --- log + PNG ---
        if step % args_cli.vis_interval == 0:
            nz = int((fn_grid > 1e-6).sum())
            print(f"[tactile_demo] step={step:5d}  "
                  f"fn max={fn_grid.max():.4f}  mean={fn_grid.mean():.5f}  "
                  f"nonzero={nz}/384")
            png = _save_heatmap(fn_grid, step, out_dir)
            print(f"               → {png}")

        step += 1
        t    += sim_dt

    print(f"\n[tactile_demo] done — {step} steps.")
    print(f"[tactile_demo] PNG heatmaps saved to: {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
