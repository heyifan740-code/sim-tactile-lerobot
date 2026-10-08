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
# Taxel layer offset along the pad normal (visually calibrated). Objects stop at the jaw collider,
# ~0.2-0.5 mm above the taxels (measured).
DEFAULT_TACTILE_NORMAL_OFFSET_M = -0.001
DEFAULT_TACTILE_DEBUG_POINT_RADIUS_M = 0.5 * DEFAULT_TACTILE_POINT_DISTANCE_M
DEFAULT_TACTILE_IDLE_BOX_POS_W = (10.0, 10.0, 10.0)
# Raw counts: clamp(COUNTS_PER_M * (SHELL - d), 0, 255), d = unsigned taxel-to-object distance.
# Equivalent to the upstream penetration rule for a taxel layer lifted just above the collider.
# A resting contact (d ~ 0.4 mm) gives ~125 counts (threshold 25 + full scale 100 of flexitac);
# the threshold is reached at d ~ 1.28 mm, i.e. within ~0.9 mm of actual contact.
DEFAULT_TACTILE_SHELL_M = 0.0015
DEFAULT_TACTILE_COUNTS_PER_M = 125.0 / (DEFAULT_TACTILE_SHELL_M - 0.0004)
DEFAULT_TACTILE_MAX_COUNTS = 255.0  # uint8 ADC range of the real sensor

if TYPE_CHECKING:
    from .config_isaacsim_so100_tactile_follower import SimTactilePadConfig


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


def build_tactile_pad_sensor(
    pad_cfg: "SimTactilePadConfig",
    arm_prim_path: str,
    query_target_prim_paths: list[str],
    table_size_m: tuple[float, float, float] = (0.8, 0.6, 0.72),
):
    """Build the flat 12x32 tactile patch on wrist_roll_tactile_pad_link.

    Uses the upstream FlexiTac ``WarpSdfTactileSensor`` unmodified. That sensor
    queries a single target mesh, so one sensor is built per query target
    (Table / Hole / Peg) and the caller combines them with a per-taxel max.
    Since fn is monotonically decreasing in the taxel-to-surface distance,
    max over targets == fn of the closest target.

    The taxel grid comes from the sensor's own generator: ``point_distance``
    is the 28 mm / 12 row pitch, and ``pad_offset`` / ``pad_quat`` place the
    patch on the pad link. Returns a list of sensors sharing the same taxels.
    """
    from .warp_sdf_tactile import WarpSdfTactileSensor, WarpSdfTactileSensorCfg

    resolved_target_mesh_paths = resolve_query_target_mesh_prim_paths(query_target_prim_paths)
    if resolved_target_mesh_paths:
        print(f"[tactile] Registered query target meshes: {resolved_target_mesh_paths}")
    else:
        print("[tactile] No mesh query targets resolved; falling back to the idle box target.")
    link_name = (pad_cfg.link_name or DEFAULT_TACTILE_PAD_LINK_NAME).strip("/")
    elastomer_prim = f"{arm_prim_path}/{link_name}"
    point_distance = _resolve_point_distance_m(pad_cfg)
    _, _, _ = table_size_m

    sensors = []
    for i, target_mesh_prim_path in enumerate(resolved_target_mesh_paths or [None]):
        cfg = WarpSdfTactileSensorCfg(
            prim_path=arm_prim_path,
            update_period=0,
            elastomer_prim_paths=[elastomer_prim],
            # 12 x 32 grid, isotropic pitch = 28 mm / 12, centred on pad_offset.
            num_rows=pad_cfg.shape[0],
            num_cols=pad_cfg.shape[1],
            point_distance=point_distance,
            normal_axis=DEFAULT_TACTILE_NORMAL_AXIS,
            normal_offset=DEFAULT_TACTILE_NORMAL_OFFSET_M,
            patch_offset_pos_b=tuple(float(v) for v in pad_cfg.pad_offset),
            patch_offset_quat_b=tuple(float(v) for v in pad_cfg.pad_quat),
            target_mesh_prim_path=target_mesh_prim_path,
            # penetration = shell - |sdf| (see DEFAULT_TACTILE_COUNTS_PER_M)
            mesh_use_signed_distance=False,
            mesh_signed_distance_method="normal",   # unused in unsigned mode
            mesh_shell_thickness=DEFAULT_TACTILE_SHELL_M,
            # No resolvable mesh query target — fall back to far-away idle box to avoid false triggers.
            box_pos_w=DEFAULT_TACTILE_IDLE_BOX_POS_W,
            box_quat_w=(1.0, 0.0, 0.0, 0.0),
            box_half_extents=(0.001, 0.001, 0.001),
            # fn = raw counts; the follower applies the flexitac normalization.
            stiffness=DEFAULT_TACTILE_COUNTS_PER_M,
            max_force=DEFAULT_TACTILE_MAX_COUNTS,
            normalize_forces=False,
            # Viewport markers: sensor 0 draws all taxels (blue), the others only contacts (red,
            # slightly larger so they render on top).
            debug_vis=True,
            debug_vis_show_all_taxels=(i == 0),
            debug_vis_show_axes=False,
            debug_vis_axes_scale=0.025,
            debug_vis_point_radius=DEFAULT_TACTILE_DEBUG_POINT_RADIUS_M * (1.0 + 0.05 * i),
        )
        sensors.append(WarpSdfTactileSensor(cfg=cfg))
    return sensors
