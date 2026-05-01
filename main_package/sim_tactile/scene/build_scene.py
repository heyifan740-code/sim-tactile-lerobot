"""Scene construction for the SO100 tactile insertion env.

Scene layout:
    ground plane
    + table (CuboidCfg)
        + SO100 arm (USD, fix_base=True) at table top centre
        + [optional] hole fixture (static rigid body)
    + [optional] peg (free rigid body)
    + top-down camera (fixed world position)
    + dome light

Returns (sim, arm, cameras) so the follower class owns all handles.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..follower.isaacsim_follower_config import IsaacSimSO100TactileFollowerConfig

_REPO = Path(__file__).resolve().parents[3]  # workspace/IsaacLab


def _spawn_hollow_hole_ring(
    prim_path: str,
    *,
    outer_diameter_m: float,
    wall_thickness_m: float,
    height_m: float,
    translation: tuple[float, float, float],
    color_rgb: tuple[float, float, float],
    rigid_props,
    collision_props,
    physics_material,
):
    import isaacsim.core.utils.prims as prim_utils
    import trimesh
    from pxr import UsdPhysics

    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh

    outer_radius_m = 0.5 * outer_diameter_m
    inner_radius_m = outer_radius_m - wall_thickness_m
    if inner_radius_m <= 0.0:
        raise ValueError(
            f"Invalid hole ring dimensions: outer_diameter_m={outer_diameter_m}, "
            f"wall_thickness_m={wall_thickness_m}."
        )

    ring_mesh = trimesh.creation.annulus(
        r_min=inner_radius_m,
        r_max=outer_radius_m,
        height=height_m,
        sections=64,
    )
    ring_cfg = sim_utils.MeshCylinderCfg(
        radius=outer_radius_m,
        height=height_m,
        axis="Z",
        rigid_props=rigid_props,
        collision_props=collision_props,
        physics_material=physics_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_rgb),
    )
    _spawn_mesh_geom_from_mesh(prim_path, ring_cfg, ring_mesh, translation=translation)

    mesh_prim = prim_utils.get_prim_at_path(f"{prim_path}/geometry/mesh")
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")


def build_scene(cfg: "IsaacSimSO100TactileFollowerConfig"):
    """Build table + arm + cameras.  Returns (sim, arm, cameras_dict).

    Must be called AFTER AppLauncher.app is started (i.e. inside connect()).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab.sim import SimulationContext, SimulationCfg

    # ------------------------------------------------------------------
    # Simulation context
    # ------------------------------------------------------------------
    sim = SimulationContext(
        SimulationCfg(
            dt=cfg.sim_dt,
            device=cfg.device,
            physx=sim_utils.PhysxCfg(
                enable_ccd=cfg.enable_ccd,
                enable_enhanced_determinism=cfg.enable_enhanced_determinism,
                solve_articulation_contact_last=cfg.solve_articulation_contact_last,
            ),
        )
    )

    # ------------------------------------------------------------------
    # World prims
    # ------------------------------------------------------------------
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0)
    )

    # Table — static cuboid, top surface at z = table_height
    tx, ty, tz = cfg.table_size_m
    table_collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=cfg.table_contact_offset_m,
        rest_offset=cfg.table_rest_offset_m,
    )
    table_physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.1,
        dynamic_friction=0.95,
        restitution=0.0,
    )
    table_cfg = sim_utils.MeshCuboidCfg(
        size=(tx, ty, tz),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        collision_props=table_collision_props,
        physics_material=table_physics_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
    )
    table_cfg.func("/World/Table", table_cfg, translation=(0.0, 0.0, tz * 0.5))

    # ------------------------------------------------------------------
    # Arm (fixed base, USD already converted)
    # ------------------------------------------------------------------
    usd_path = cfg.usd_path or str(
        _REPO / "main_package/sim_tactile/assets/gripper_so100_tactile/usd/so100_follower_tactile.usd"
    )

    bx, by, bz = cfg.arm_base_pos_m
    bw, bi, bj, bk = cfg.arm_base_quat  # (w, x, y, z)

    arm_cfg = ArticulationCfg(
        prim_path="/World/Arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=cfg.arm_max_depenetration_velocity,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=cfg.arm_solver_position_iteration_count,
                solver_velocity_iteration_count=cfg.arm_solver_velocity_iteration_count,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cfg.arm_contact_offset_m,
                rest_offset=cfg.arm_rest_offset_m,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=cfg.enable_self_collisions,
                fix_root_link=True,
                solver_position_iteration_count=cfg.arm_solver_position_iteration_count,
                solver_velocity_iteration_count=cfg.arm_solver_velocity_iteration_count,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(bx, by, bz),
            rot=(bw, bi, bj, bk),
            joint_pos={
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        ),
        actuators={
            "arm_actuators": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex",
                                  "wrist_flex", "wrist_roll", "gripper"],
                stiffness=200.0,
                damping=10.0,
                effort_limit=10.0,
            ),
        },
    )
    arm = Articulation(cfg=arm_cfg)

    # ------------------------------------------------------------------
    # Optional hole fixture (static) and graspable peg.
    # ------------------------------------------------------------------
    if cfg.spawn_task_objects:
        hx, hy, hz = cfg.hole_init_pos_m
        px, py, pz = cfg.peg_init_pos_m

        hole_collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=min(cfg.peg_contact_offset_m, max(0.0005, 0.5 * cfg.hole_wall_thickness_m)),
            rest_offset=0.0,
        )

        if cfg.hole_stl_path:
            hole_cfg = sim_utils.MeshCfg(
                visual_mesh_file_path=cfg.hole_stl_path,
                collision_mesh_file_path=cfg.hole_stl_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=hole_collision_props,
                physics_material=table_physics_material,
            )
            hole_cfg.func(
                "/World/Hole",
                hole_cfg,
                translation=(hx, hy, hz),
            )
        else:
            _spawn_hollow_hole_ring(
                "/World/Hole",
                outer_diameter_m=cfg.hole_diameter_m,
                wall_thickness_m=cfg.hole_wall_thickness_m,
                height_m=cfg.hole_height_m,
                translation=(hx, hy, hz),
                color_rgb=cfg.hole_color_rgb,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=hole_collision_props,
                physics_material=table_physics_material,
            )

        peg_collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=cfg.peg_contact_offset_m,
            rest_offset=cfg.peg_rest_offset_m,
        )
        peg_physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        peg_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=cfg.arm_max_depenetration_velocity,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        )

        if cfg.peg_stl_path:
            peg_cfg = sim_utils.MeshCfg(
                visual_mesh_file_path=cfg.peg_stl_path,
                collision_mesh_file_path=cfg.peg_stl_path,
                rigid_props=peg_rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=cfg.peg_mass_kg),
                collision_props=peg_collision_props,
                physics_material=peg_physics_material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.peg_color_rgb),
            )
            peg_cfg.func(
                "/World/Peg",
                peg_cfg,
                translation=(px, py, pz),
            )
        else:
            peg_cfg = sim_utils.MeshCylinderCfg(
                radius=0.5 * cfg.peg_diameter_m,
                height=cfg.peg_height_m,
                axis="Z",
                rigid_props=peg_rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=cfg.peg_mass_kg),
                collision_props=peg_collision_props,
                physics_material=peg_physics_material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.peg_color_rgb),
            )
            peg_cfg.func(
                "/World/Peg",
                peg_cfg,
                translation=(px, py, pz),
            )

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    # Fixed top-down / angled view camera — good enough for collecting demos.
    # Wrist camera can be added by attaching a CameraCfg to gripper_frame_link.
    cameras: dict = {}

    # Top camera: looking down at the workspace from above and slightly behind.
    top_cam_cfg = CameraCfg(
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
            pos=(0.8, 0.0, 1.4),
            rot=(0.0, -0.3826834, 0.0, 0.9238795),  # look -x, tilt down
            convention="world",
        ),
    )
    cameras["top"] = Camera(cfg=top_cam_cfg)

    sim.set_camera_view(eye=(0.0, -1.2, 1.0), target=(0.0, 0.0, 0.7))

    return sim, arm, cameras
