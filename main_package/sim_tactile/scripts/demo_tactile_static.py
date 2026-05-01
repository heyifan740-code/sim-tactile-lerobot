"""静态 SO100 + 可视化 tactile pad（3D taxel 球 + 2D 热力图面板）

设计要点：
    - 直接复用 URDF 里的 wrist_roll_tactile_pad_link 作为 tactile 锚点
    - 默认使用手动平面贴片，方便直接调 offset 对齐目标区域
    - 默认不再启用 auto-follow 假接触盒，先只看点位是否贴面

可视化：
  1. IsaacSim 3D 视口：彩色 taxel 球（红=接触，蓝=无接触）+ XYZ 轴 marker
  2. docked omni.ui 窗口：12×32 viridis 热力图，每帧更新

用法：
    ./isaaclab.sh -p main_package/sim_tactile/scripts/demo_tactile_static.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO100 静态 tactile pad 可视化")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--max-steps", type=int, default=0)
parser.add_argument("--vis-scale", type=int, default=12, help="每个 taxel 的像素放大倍数")
parser.add_argument("--auto-contact-demo", action="store_true", help="启用旧的 auto-follow 解析盒，人工制造接触信号")
parser.add_argument("--check-host-penetration", action="store_true", help="用 Wrist_Roll_08c 宿主 mesh 检查 taxel 穿透是否产生 fn")
parser.add_argument("--pad-offset-x-mm", type=float, default=0.0, help="沿 pad 局部 X 平移贴片，单位 mm")
parser.add_argument("--pad-offset-y-mm", type=float, default=0.0, help="沿 pad 局部 Y 平移贴片，单位 mm")
parser.add_argument("--pad-offset-z-mm", type=float, default=0.0, help="沿 pad 局部 Z 平移贴片，单位 mm")
parser.add_argument("--pad-rot-z-deg", type=float, default=0.0, help="绕 pad 局部 Z 轴旋转贴片，单位 deg")
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = False   # omni.ui 热力图面板需要 GUI

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 在 AppLauncher 之后导入 ───────────────────────────────────────────────
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors.warp_sdf_tactile import WarpSdfTactileSensor, WarpSdfTactileSensorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from main_package.sim_tactile.sensors.pad_on_wrist_roll import (
    DEFAULT_TACTILE_DEBUG_POINT_RADIUS_M,
    DEFAULT_TACTILE_IDLE_BOX_POS_W,
    DEFAULT_TACTILE_NORMAL_AXIS,
    DEFAULT_TACTILE_NORMAL_OFFSET_M,
    DEFAULT_TACTILE_PAD_LINK_NAME,
    DEFAULT_TACTILE_PATCH_OFFSET_POS_B,
    DEFAULT_TACTILE_PATCH_OFFSET_QUAT_B,
    DEFAULT_TACTILE_POINT_DISTANCE_M,
    DEFAULT_TACTILE_TARGET_LENGTH_M,
    DEFAULT_TACTILE_TARGET_WIDTH_M,
    create_manual_tactile_pad_points_local,
    resolve_tactile_target_mesh_prim_path,
)

_REPO = Path(__file__).resolve().parents[3]
_USD  = _REPO / "main_package/sim_tactile/assets/gripper_so100_tactile/usd/so100_follower_tactile.usd"

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# ── 传感器几何参数（与 pad_on_wrist_roll.py 保持一致）────────────────────
_PAD_LINK = DEFAULT_TACTILE_PAD_LINK_NAME
# auto-follow 解析盒尺寸：比 taxel grid 略小 → 出现边缘 fn=0，内部 fn>0
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


# ── omni.ui 热力图面板 ────────────────────────────────────────────────────
def _create_heatmap_panel(num_rows: int, num_cols: int, scale: int = 12):
    """在 IsaacSim 右侧创建 12×32 viridis 热力图窗口。"""
    import asyncio
    try:
        import omni.ui as ui
        import omni.kit.app
    except ImportError:
        print("[热力图] omni.ui 不可用，跳过面板。")
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
            ui.Label("Wrist_Roll_08c 内侧面   12×32 taxels   viridis (0→1)")
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
    print(f"[热力图] 已创建窗口 '{window.title}'  ({w}×{h} px)")

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


# ── 主程序 ────────────────────────────────────────────────────────────────
def main() -> None:
    sim_dt = 1.0 / 60.0
    sim = SimulationContext(SimulationCfg(dt=sim_dt, device=args_cli.device))

    # ── 世界基础 prims ───────────────────────────────────────────────────
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

    # ── 机械臂 ───────────────────────────────────────────────────────────
    arm = Articulation(cfg=ArticulationCfg(
        prim_path="/World/Arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_USD),
            activate_contact_sensors=True,   # ContactSensor pose 追踪必须
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
    local_points_b = create_manual_tactile_pad_points_local(
        num_rows=12,
        num_cols=32,
        row_center_span_m=DEFAULT_TACTILE_TARGET_WIDTH_M,
        col_center_span_m=DEFAULT_TACTILE_TARGET_LENGTH_M,
        normal_offset_m=DEFAULT_TACTILE_NORMAL_OFFSET_M,
        patch_offset_pos_b=pad_offset_b,
        patch_offset_quat_b=pad_quat_b,
    )
    row_pitch_m = DEFAULT_TACTILE_TARGET_WIDTH_M / 12.0
    col_pitch_m = DEFAULT_TACTILE_TARGET_LENGTH_M / 32.0
    target_mesh_prim_path = None
    if args_cli.check_host_penetration:
        target_mesh_prim_path = resolve_tactile_target_mesh_prim_path(arm_prim_path="/World/Arm")

    # ── 触觉传感器（手动平面贴片；宽边按 2.8 cm 目标推导）───────────────
    tactile = WarpSdfTactileSensor(cfg=WarpSdfTactileSensorCfg(
        prim_path="/World/Arm",
        elastomer_prim_paths=[f"/World/Arm/{_PAD_LINK}"],
        local_points_b_per_elastomer=[local_points_b],
        target_mesh_prim_path=target_mesh_prim_path,
        # Unsigned shell mode：方向对称、不要求 mesh watertight
        mesh_use_signed_distance=False,
        mesh_signed_distance_method="normal",
        mesh_shell_thickness=0.005,             # 5 mm 接触感应范围
        mesh_contact_onset_m=0.0,
        num_rows=12,
        num_cols=32,
        point_distance=min(DEFAULT_TACTILE_POINT_DISTANCE_M, col_pitch_m),
        normal_axis=DEFAULT_TACTILE_NORMAL_AXIS,
        normal_offset=0.0,
        patch_offset_pos_b=DEFAULT_TACTILE_PATCH_OFFSET_POS_B,
        patch_offset_quat_b=DEFAULT_TACTILE_PATCH_OFFSET_QUAT_B,
        # 默认无接触；仅在 --auto-contact-demo 下启用旧的解析盒演示。
        box_pos_w=DEFAULT_TACTILE_IDLE_BOX_POS_W,
        box_quat_w=(1.0, 0.0, 0.0, 0.0),
        box_half_extents=_BOX_HALF,
        stiffness=500.0,
        max_force=1.0,
        normalize_forces=True,
        # 3D taxel 球体 marker（轴关掉，避免遮挡接触视线）
        debug_vis=True,
        debug_vis_show_all_taxels=True,     # 蓝=无接触，红=有接触
        debug_vis_show_axes=False,
        debug_vis_axes_scale=0.025,
        debug_vis_point_radius=DEFAULT_TACTILE_DEBUG_POINT_RADIUS_M,
    ))

    # ── reset ─────────────────────────────────────────────────────────────
    sim.reset()
    arm.reset()
    tactile.reset()
    sim.set_camera_view(eye=(0.20, -0.30, 1.0), target=(0.02, 0.0, 0.85))

    # ── body 索引 ─────────────────────────────────────────────────────────
    joint_names = list(arm.data.joint_names)
    joint_idx   = {n: i for i, n in enumerate(joint_names)}
    body_names  = list(arm.data.body_names)
    print(f"\n[demo] 关节: {joint_names}")
    print(f"[demo] 刚体: {body_names}")

    if _PAD_LINK in body_names:
        pad_idx = body_names.index(_PAD_LINK)
        print(f"[demo] {_PAD_LINK} body_idx={pad_idx} ✓")
    else:
        pad_idx = None
        print(f"[demo] 警告：body_names 中未找到 {_PAD_LINK}！")

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

    # ── 2D 热力图面板 ─────────────────────────────────────────────────────
    panel = _create_heatmap_panel(num_rows=12, num_cols=32, scale=args_cli.vis_scale)
    if panel is None:
        print("[demo] 无热力图面板，仅依赖 3D taxel 球体可视化。")

    target = arm.data.joint_pos.clone()   # 静止在零位
    step   = 0
    max_steps = args_cli.max_steps

    print("[demo] 静止展示 — 在视口中旋转观察 taxel 球体位置。")
    if args_cli.auto_contact_demo:
        print("[demo] 已启用 auto-contact-demo：解析盒会跟踪 pad，红点表示人工生成的接触。\n")
    elif args_cli.check_host_penetration:
        print("[demo] 已启用 host-penetration check：若 taxel 点进入 Wrist_Roll_08c 宿主 mesh，应出现 fn>0。\n")
    else:
        print("[demo] 默认只检查贴面位置：无 auto-follow 假接触，taxel 应主要显示为蓝色。\n")

    # ── 主循环 ────────────────────────────────────────────────────────────
    while simulation_app.is_running() and (max_steps == 0 or step < max_steps):
        # 零位保持
        arm.set_joint_position_target(target)
        arm.write_data_to_sim()
        sim.step()
        arm.update(sim_dt)

        # auto-follow：仅在显式请求时，才让解析盒跟踪 pad link 世界位姿
        if args_cli.auto_contact_demo and pad_idx is not None:
            # body_state_w: [E, B, 13]  pos=[:3]  quat(w,x,y,z)=[3:7]
            pad_pos_w = arm.data.body_state_w[0, pad_idx, :3]
            pad_quat_w = arm.data.body_state_w[0, pad_idx, 3:7]
            tactile.set_box_pose(pad_pos_w, pad_quat_w)

        # 更新传感器
        tactile.update(dt=sim_dt, force_recompute=True)

        # 读取 fn  (E=1, S=1, P=384, 4)
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

    print(f"\n[demo] 完成 — {step} 帧。")


if __name__ == "__main__":
    main()
    simulation_app.close()
