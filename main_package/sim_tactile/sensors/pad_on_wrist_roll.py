"""Tactile pad mounted on Wrist_Roll_08c's inner face.

URDF/ USD already defines ``wrist_roll_tactile_pad_link`` as a fixed child of
``gripper_link`` at the pad surface centre. Reusing that link as the tactile
frame keeps the taxel patch aligned with the converted asset instead of
duplicating a second hand-tuned offset in Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

DEFAULT_TACTILE_PAD_LINK_NAME = "wrist_roll_tactile_pad_link"
DEFAULT_TACTILE_PROJECTION_MESH_LINK_NAME = "gripper_link"
DEFAULT_TACTILE_PROJECTION_MESH_NAME_HINTS = ("wrist_roll_08c", "wrist_roll")
DEFAULT_TACTILE_PATCH_OFFSET_POS_B = (0.019, 0.0, -0.01072)
DEFAULT_TACTILE_PATCH_OFFSET_QUAT_B = (0.7071067811865476, 0.0, 0.0, -0.7071067811865475)
DEFAULT_TACTILE_TARGET_WIDTH_M = 0.028
DEFAULT_TACTILE_TARGET_WIDTH_POINT_COUNT = 12
DEFAULT_TACTILE_POINT_DISTANCE_M = DEFAULT_TACTILE_TARGET_WIDTH_M / float(DEFAULT_TACTILE_TARGET_WIDTH_POINT_COUNT)
DEFAULT_TACTILE_TARGET_LENGTH_M = DEFAULT_TACTILE_POINT_DISTANCE_M * 32.0
DEFAULT_TACTILE_NORMAL_AXIS = 2
# Keep the visible pad pose fixed and place the hidden taxel sampling layer on
# the tactile surface. Negative offset sinks taxels slightly INTO the pad so
# they sit flush with the gripper jaw inner surface (manually calibrated).
DEFAULT_TACTILE_NORMAL_OFFSET_M = -0.001
DEFAULT_TACTILE_DEBUG_POINT_RADIUS_M = 0.5 * DEFAULT_TACTILE_POINT_DISTANCE_M
DEFAULT_TACTILE_IDLE_BOX_POS_W = (10.0, 10.0, 10.0)
DEFAULT_TACTILE_SURFACE_RAY_START_OFFSET_M = 0.01
DEFAULT_TACTILE_SURFACE_NORMAL_MIN_Z = 0.25

if TYPE_CHECKING:
    from ..follower.isaacsim_follower_config import SimTactilePadConfig


def _resolve_point_distance_m(pad_cfg: "SimTactilePadConfig") -> float:
    rows, cols = int(pad_cfg.shape[0]), int(pad_cfg.shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Tactile pad shape must be positive, got {pad_cfg.shape}.")

    row_pitch = float(pad_cfg.grid_size_m[0]) / float(rows)
    col_pitch = float(pad_cfg.grid_size_m[1]) / float(cols)
    if row_pitch <= 0.0 or col_pitch <= 0.0:
        raise ValueError(f"Tactile pad grid_size_m must be positive, got {pad_cfg.grid_size_m}.")
    if abs(row_pitch - col_pitch) > 1.0e-6:
        raise ValueError(
            "WarpSdfTactileSensor currently expects isotropic taxel spacing, "
            f"got row_pitch={row_pitch:.6f} m and col_pitch={col_pitch:.6f} m."
        )
    return row_pitch


def _triangulate_face_vertex_indices(face_counts, face_indices) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    idx = 0
    for count in face_counts:
        n = int(count)
        if n < 3:
            idx += n
            continue
        v0 = int(face_indices[idx])
        for i in range(1, n - 1):
            triangles.append((v0, int(face_indices[idx + i]), int(face_indices[idx + i + 1])))
        idx += n
    return triangles


def _apply_patch_transform_to_points(
    points_b: list[tuple[float, float, float]],
    patch_offset_pos_b: tuple[float, float, float],
    patch_offset_quat_b: tuple[float, float, float, float],
) -> list[tuple[float, float, float]]:
    import torch

    from isaaclab.utils.math import quat_apply

    if not points_b:
        return []

    pts_t = torch.tensor(points_b, dtype=torch.float32)
    pos_t = torch.tensor(patch_offset_pos_b, dtype=torch.float32)
    quat_t = torch.tensor(patch_offset_quat_b, dtype=torch.float32).unsqueeze(0).expand(pts_t.shape[0], -1)
    transformed = quat_apply(quat_t, pts_t) + pos_t
    return [tuple(float(v) for v in point) for point in transformed.tolist()]


def create_manual_tactile_pad_points_local(
    *,
    num_rows: int,
    num_cols: int,
    row_center_span_m: float,
    col_center_span_m: float,
    normal_offset_m: float = DEFAULT_TACTILE_NORMAL_OFFSET_M,
    patch_offset_pos_b: tuple[float, float, float] = DEFAULT_TACTILE_PATCH_OFFSET_POS_B,
    patch_offset_quat_b: tuple[float, float, float, float] = DEFAULT_TACTILE_PATCH_OFFSET_QUAT_B,
) -> list[tuple[float, float, float]]:
    import numpy as np

    if num_rows <= 0 or num_cols <= 0:
        raise ValueError(f"Manual tactile patch shape must be positive, got ({num_rows}, {num_cols}).")
    if row_center_span_m <= 0.0 or col_center_span_m <= 0.0:
        raise ValueError(
            "Manual tactile patch spans must be positive, "
            f"got row_center_span_m={row_center_span_m}, col_center_span_m={col_center_span_m}."
        )

    row_cell_m = float(row_center_span_m) / float(num_rows)
    col_cell_m = float(col_center_span_m) / float(num_cols)

    u = np.linspace(
        -0.5 * float(row_center_span_m) + 0.5 * row_cell_m,
        +0.5 * float(row_center_span_m) - 0.5 * row_cell_m,
        int(num_rows),
        dtype=np.float32,
    )
    v = np.linspace(
        -0.5 * float(col_center_span_m) + 0.5 * col_cell_m,
        +0.5 * float(col_center_span_m) - 0.5 * col_cell_m,
        int(num_cols),
        dtype=np.float32,
    )
    uu, vv = np.meshgrid(u, v, indexing="ij")
    points_b = np.stack(
        (
            uu.reshape(-1),
            vv.reshape(-1),
            np.full((int(num_rows) * int(num_cols),), float(normal_offset_m), dtype=np.float32),
        ),
        axis=1,
    )
    return _apply_patch_transform_to_points(
        [tuple(float(v) for v in point) for point in points_b.tolist()],
        patch_offset_pos_b=patch_offset_pos_b,
        patch_offset_quat_b=patch_offset_quat_b,
    )


def _make_prim_search_text(prim) -> str:
    parts: list[str] = []
    current = prim
    while current and current.IsValid():
        parts.append(current.GetName())
        parts.append(str(current.GetPath()))
        current = current.GetParent()
    return " ".join(parts).lower()


def _select_projection_mesh_prims(mesh_prims, name_hints: tuple[str, ...]):
    normalized_hints = tuple(hint.strip().lower() for hint in name_hints if hint.strip())
    if not normalized_hints:
        return list(mesh_prims), False

    selected = []
    for mesh_prim in mesh_prims:
        search_text = _make_prim_search_text(mesh_prim)
        if any(hint in search_text for hint in normalized_hints):
            selected.append(mesh_prim)
    return (selected if selected else list(mesh_prims), bool(selected))


def resolve_tactile_target_mesh_prim_path(
    *,
    arm_prim_path: str,
    projection_mesh_link_name: str = DEFAULT_TACTILE_PROJECTION_MESH_LINK_NAME,
    projection_mesh_name_hints: tuple[str, ...] = DEFAULT_TACTILE_PROJECTION_MESH_NAME_HINTS,
) -> str:
    import omni.usd

    import isaaclab.sim as sim_utils

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available for tactile target mesh resolution.")

    mesh_root_path = f"{arm_prim_path}/{projection_mesh_link_name}"
    mesh_prims = sim_utils.get_all_matching_child_prims(
        mesh_root_path,
        predicate=lambda prim: prim.GetTypeName() == "Mesh",
    )
    if not mesh_prims:
        raise RuntimeError(f"No mesh prims found under tactile target mesh root: {mesh_root_path}")

    mesh_prims, _ = _select_projection_mesh_prims(mesh_prims, projection_mesh_name_hints)
    mesh_paths = [str(prim.GetPath()) for prim in mesh_prims]
    collision_paths = [path for path in mesh_paths if "/collisions/" in path]
    preferred_paths = collision_paths if collision_paths else mesh_paths
    return preferred_paths[0]


def resolve_query_target_mesh_prim_paths(query_target_prim_paths: list[str]) -> list[str]:
    import omni.usd

    import isaaclab.sim as sim_utils

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available for tactile target mesh resolution.")

    resolved_paths: list[str] = []
    for prim_path in query_target_prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[tactile] WARNING: query target prim is invalid and will be skipped: {prim_path}")
            continue

        if prim.GetTypeName() == "Mesh":
            resolved_paths.append(str(prim.GetPath()))
            continue

        mesh_prims = sim_utils.get_all_matching_child_prims(
            prim_path,
            predicate=lambda child_prim: child_prim.GetTypeName() == "Mesh",
        )
        if not mesh_prims:
            print(f"[tactile] WARNING: no mesh prim found under tactile query target: {prim_path}")
            continue

        mesh_paths = [str(mesh_prim.GetPath()) for mesh_prim in mesh_prims]
        preferred_paths = [path for path in mesh_paths if path.endswith("/geometry/mesh")]
        if not preferred_paths:
            preferred_paths = [path for path in mesh_paths if "/collisions/" in path]
        if not preferred_paths:
            preferred_paths = mesh_paths
        resolved_paths.append(preferred_paths[0])

    return resolved_paths


def project_tactile_pad_points_local(
    *,
    arm_prim_path: str,
    num_rows: int,
    num_cols: int,
    point_distance: float,
    pad_link_name: str = DEFAULT_TACTILE_PAD_LINK_NAME,
    projection_mesh_link_name: str = DEFAULT_TACTILE_PROJECTION_MESH_LINK_NAME,
    projection_mesh_name_hints: tuple[str, ...] = DEFAULT_TACTILE_PROJECTION_MESH_NAME_HINTS,
    normal_offset_m: float = DEFAULT_TACTILE_NORMAL_OFFSET_M,
    patch_offset_pos_b: tuple[float, float, float] = DEFAULT_TACTILE_PATCH_OFFSET_POS_B,
    patch_offset_quat_b: tuple[float, float, float, float] = DEFAULT_TACTILE_PATCH_OFFSET_QUAT_B,
    ray_start_offset_m: float = DEFAULT_TACTILE_SURFACE_RAY_START_OFFSET_M,
) -> list[tuple[float, float, float]]:
    """Project a regular taxel grid onto the inner jaw mesh in the pad-link frame.

    This mirrors FlexiTac's mesh-raycast approach: create a regular XY grid in the
    tactile frame, cast rays along +Z into the jaw mesh, then place each taxel 1 mm
    inside the elastomer along the local normal.
    """
    import numpy as np
    import omni.usd
    import trimesh
    from pxr import Gf, UsdGeom

    import isaaclab.sim as sim_utils

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available for tactile point projection.")

    pad_prim = stage.GetPrimAtPath(f"{arm_prim_path}/{pad_link_name}")
    if not pad_prim.IsValid():
        raise RuntimeError(f"Invalid pad-link prim for tactile projection: {arm_prim_path}/{pad_link_name}")

    mesh_root_path = f"{arm_prim_path}/{projection_mesh_link_name}"
    mesh_prims = sim_utils.get_all_matching_child_prims(
        mesh_root_path,
        predicate=lambda prim: prim.GetTypeName() == "Mesh",
    )
    if not mesh_prims:
        raise RuntimeError(f"No mesh prims found under tactile projection root: {mesh_root_path}")

    mesh_prims, used_name_filter = _select_projection_mesh_prims(mesh_prims, projection_mesh_name_hints)
    if used_name_filter:
        print(
            "[tactile] Projection mesh filter matched "
            f"{len(mesh_prims)} prim(s): {', '.join(str(prim.GetPath()) for prim in mesh_prims)}"
        )
    else:
        print(
            "[tactile] WARNING: projection mesh name filter did not match any prims under "
            f"{mesh_root_path}; falling back to all meshes."
        )

    xform_cache = UsdGeom.XformCache()
    world_to_pad = xform_cache.GetLocalToWorldTransform(pad_prim).GetInverse()

    vertices_pad: list[np.ndarray] = []
    faces_pad: list[np.ndarray] = []
    vertex_offset = 0
    for mesh_prim in mesh_prims:
        mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if points.size == 0 or face_counts.size == 0 or face_indices.size == 0:
            continue

        if np.all(face_counts == 3) and (face_indices.size % 3 == 0):
            triangles = face_indices.reshape(-1, 3)
        else:
            triangles = np.asarray(_triangulate_face_vertex_indices(face_counts, face_indices), dtype=np.int32)
        if triangles.size == 0:
            continue

        mesh_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
        transformed = np.empty_like(points, dtype=np.float32)
        for idx, point in enumerate(points):
            point_world = mesh_to_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
            point_pad = world_to_pad.Transform(point_world)
            transformed[idx] = (float(point_pad[0]), float(point_pad[1]), float(point_pad[2]))

        vertices_pad.append(transformed)
        faces_pad.append(triangles + vertex_offset)
        vertex_offset += transformed.shape[0]

    if not vertices_pad or not faces_pad:
        raise RuntimeError(f"Failed to build tactile projection mesh from {mesh_root_path}")

    mesh_tm = trimesh.Trimesh(
        vertices=np.concatenate(vertices_pad, axis=0),
        faces=np.concatenate(faces_pad, axis=0),
        process=False,
    )
    face_mask = mesh_tm.face_normals[:, 2] >= float(DEFAULT_TACTILE_SURFACE_NORMAL_MIN_Z)
    if np.any(face_mask):
        num_face_candidates = int(np.count_nonzero(face_mask))
        if num_face_candidates != int(mesh_tm.faces.shape[0]):
            mesh_tm = trimesh.Trimesh(vertices=mesh_tm.vertices.copy(), faces=mesh_tm.faces[face_mask], process=False)
            print(
                "[tactile] Filtered projection surface faces by +Z normal: "
                f"{num_face_candidates}/{int(face_mask.shape[0])} faces kept."
            )
    else:
        print(
            "[tactile] WARNING: +Z face-normal filtering kept 0 faces; "
            "falling back to all faces for projection."
        )

    u = np.linspace(
        -point_distance * (num_rows + 1) / 2.0,
        +point_distance * (num_rows + 1) / 2.0,
        num_rows + 2,
        dtype=np.float32,
    )[1:-1]
    v = np.linspace(
        -point_distance * (num_cols + 1) / 2.0,
        +point_distance * (num_cols + 1) / 2.0,
        num_cols + 2,
        dtype=np.float32,
    )[1:-1]
    uu, vv = np.meshgrid(u, v, indexing="ij")
    num_points = int(num_rows) * int(num_cols)

    ray_start_z = float(np.min(mesh_tm.vertices[:, 2])) - float(ray_start_offset_m)
    ray_origins = np.stack(
        (uu.reshape(-1), vv.reshape(-1), np.full((num_points,), ray_start_z, dtype=np.float32)),
        axis=1,
    )
    ray_directions = np.zeros((num_points, 3), dtype=np.float32)
    ray_directions[:, 2] = 1.0

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_tm)
    _, ray_indices, locations = intersector.intersects_id(
        ray_origins,
        ray_directions,
        return_locations=True,
        multiple_hits=False,
    )
    ray_indices_np = np.asarray(ray_indices, dtype=np.int64)

    projected = np.stack(
        (uu.reshape(-1), vv.reshape(-1), np.zeros((num_points,), dtype=np.float32)),
        axis=1,
    )
    if len(ray_indices) > 0:
        projected[ray_indices_np] = np.asarray(locations, dtype=np.float32)
    if len(ray_indices) != num_points:
        miss_mask = np.ones((num_points,), dtype=bool)
        miss_mask[ray_indices_np] = False
        miss_indices = np.flatnonzero(miss_mask)
        try:
            nearest_points, _, _ = trimesh.proximity.closest_point(mesh_tm, projected[miss_indices])
            nearest_points = np.asarray(nearest_points, dtype=np.float32)
            valid_mask = np.all(np.isfinite(nearest_points), axis=1)
            if np.any(valid_mask):
                projected[miss_indices[valid_mask]] = nearest_points[valid_mask]
                print(
                    "[tactile] Filled ray-miss taxels from the nearest surface: "
                    f"{int(np.count_nonzero(valid_mask))}/{int(miss_indices.shape[0])}."
                )
            remaining_misses = int(miss_indices.shape[0] - np.count_nonzero(valid_mask))
            if remaining_misses > 0:
                print(
                    f"[tactile] WARNING: {remaining_misses}/{num_points} taxels still missed the inner surface; "
                    "those points fall back to the pad plane."
                )
        except Exception as exc:
            print(
                f"[tactile] WARNING: nearest-surface fallback failed ({exc}); "
                f"{int(miss_indices.shape[0])}/{num_points} taxels fall back to the pad plane."
            )

    projected[:, 2] += float(normal_offset_m)
    return _apply_patch_transform_to_points(
        [tuple(float(v) for v in point) for point in projected.tolist()],
        patch_offset_pos_b=patch_offset_pos_b,
        patch_offset_quat_b=patch_offset_quat_b,
    )


def build_tactile_pad_sensor(
    pad_cfg: "SimTactilePadConfig",
    arm_prim_path: str,
    query_target_prim_paths: list[str],
    table_size_m: tuple[float, float, float] = (0.8, 0.6, 0.72),
):
    """Build a manual flat tactile patch on wrist_roll_tactile_pad_link.

    Default geometry targets a 28 mm width across the 12-point edge,
    and then tiles the 32-point edge with the same sphere diameter for zero gaps,
    while `pad_offset` / `pad_quat` remain the knobs for manual visual tuning.
    """
    from isaaclab.sensors.warp_sdf_tactile import WarpSdfTactileSensor, WarpSdfTactileSensorCfg

    resolved_target_mesh_paths = resolve_query_target_mesh_prim_paths(query_target_prim_paths)
    if resolved_target_mesh_paths:
        print(f"[tactile] Registered query target meshes: {resolved_target_mesh_paths}")
    else:
        print("[tactile] No mesh query targets resolved; falling back to the idle box target.")
    link_name = (pad_cfg.link_name or DEFAULT_TACTILE_PAD_LINK_NAME).strip("/")
    elastomer_prim = f"{arm_prim_path}/{link_name}"
    row_center_span_m = float(pad_cfg.grid_size_m[0])
    col_center_span_m = float(pad_cfg.grid_size_m[1])
    local_points_b = create_manual_tactile_pad_points_local(
        num_rows=int(pad_cfg.shape[0]),
        num_cols=int(pad_cfg.shape[1]),
        row_center_span_m=row_center_span_m,
        col_center_span_m=col_center_span_m,
        normal_offset_m=DEFAULT_TACTILE_NORMAL_OFFSET_M,
        patch_offset_pos_b=tuple(float(v) for v in pad_cfg.pad_offset),
        patch_offset_quat_b=tuple(float(v) for v in pad_cfg.pad_quat),
    )
    row_pitch = row_center_span_m / float(max(int(pad_cfg.shape[0]), 1))
    col_pitch = col_center_span_m / float(max(int(pad_cfg.shape[1]), 1))
    point_distance = min(row_pitch, col_pitch)
    _, _, _ = table_size_m

    cfg = WarpSdfTactileSensorCfg(
        prim_path=arm_prim_path,
        update_period=0,
        elastomer_prim_paths=[elastomer_prim],
        local_points_b_per_elastomer=[local_points_b],
        # 12 × 32，默认宽边中心跨度 28 mm，长边中心跨度 32 mm。
        num_rows=pad_cfg.shape[0],
        num_cols=pad_cfg.shape[1],
        point_distance=point_distance,
        normal_axis=DEFAULT_TACTILE_NORMAL_AXIS,
        normal_offset=0.0,
        patch_offset_pos_b=DEFAULT_TACTILE_PATCH_OFFSET_POS_B,
        patch_offset_quat_b=DEFAULT_TACTILE_PATCH_OFFSET_QUAT_B,
        target_mesh_prim_path=(resolved_target_mesh_paths[0] if resolved_target_mesh_paths else None),
        target_mesh_prim_paths=(resolved_target_mesh_paths if resolved_target_mesh_paths else None),
        # 使用 UNSIGNED 距离 + shell threshold（FlexiTac 的方式不同 —— 它们用
        # winding signed，要求 mesh 严格 watertight；我们这里有 hole ring 等
        # 非闭合几何，winding 的 sign 不可靠，会出现「只有 -Y 偶发穿透」的方向
        # 偏置假象）。
        # 公式: penetration = clamp(shell - |sdf|, 0, ∞)
        # 任何方向、任何 mesh 形状都对称：当 taxel 离任一目标表面 < shell 时给出
        # 与距离线性相关的 fn 值。
        mesh_use_signed_distance=False,
        mesh_signed_distance_method="normal",   # 兜底（unsigned 模式不使用此参数）
        mesh_shell_thickness=0.005,             # 5 mm 接触感应范围
        mesh_contact_onset_m=0.0,               # unsigned 模式下不生效
        refresh_target_poses_from_stage=False,
        # 没有可解析的 mesh query target 时，回退到远处 idle box，避免误触发。
        box_pos_w=DEFAULT_TACTILE_IDLE_BOX_POS_W,
        box_quat_w=(1.0, 0.0, 0.0, 0.0),
        box_half_extents=(0.001, 0.001, 0.001),
        # 弹簧模型：sdf=0（贴面）→ penetration=5mm → fn=2.5（截断到 max_force=1）
        # sdf=2mm → penetration=3mm → fn_norm = 0.6
        # sdf=4mm → penetration=1mm → fn_norm = 0.2
        # sdf≥shell（5mm）→ fn=0
        stiffness=500.0,
        max_force=1.0,
        normalize_forces=True,
        # IsaacSim viewport: show taxel point-cloud only (axes off — they
        # visually overlap the pad/peg surfaces during teleop).
        debug_vis=True,
        debug_vis_show_all_taxels=True,
        debug_vis_show_axes=False,
        debug_vis_axes_scale=0.025,
        debug_vis_point_radius=DEFAULT_TACTILE_DEBUG_POINT_RADIUS_M,
        # Show the debug spheres at the actual taxel sample positions.
        debug_vis_normal_offset=0.0,
    )
    return WarpSdfTactileSensor(cfg=cfg)
