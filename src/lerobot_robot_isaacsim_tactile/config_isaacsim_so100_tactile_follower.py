"""Config for the IsaacSim-backed SO100 tactile follower.

Mirrors `SOTactileFollowerConfig` from lerobot_tactile but replaces
hardware-only fields (serial port, baud rate) with sim-only fields
(USD asset, scene layout, sim dt, control fps).

Registered under the draccus tag `isaacsim_so100_tactile_follower` so
that a user can point any lerobot script at this class via
`--robot.type=isaacsim_so100_tactile_follower` etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

_DEFAULT_TACTILE_TARGET_WIDTH_M = 0.028
_DEFAULT_TACTILE_TARGET_WIDTH_POINT_COUNT = 12
_DEFAULT_TACTILE_POINT_DISTANCE_M = _DEFAULT_TACTILE_TARGET_WIDTH_M / float(
    _DEFAULT_TACTILE_TARGET_WIDTH_POINT_COUNT
)
_DEFAULT_TACTILE_TARGET_LENGTH_M = _DEFAULT_TACTILE_POINT_DISTANCE_M * 32.0
_DEFAULT_TACTILE_GRID_SIZE_M = (_DEFAULT_TACTILE_TARGET_WIDTH_M, _DEFAULT_TACTILE_TARGET_LENGTH_M)


@dataclass
class SimTactilePadConfig:
    """Single tactile pad mounted on a named URDF link.

    Shape / resolution must match the physical sensor (12×32 for the
    latest real driver). Pad extents are in the mount link's local frame.
    """

    # Link name in the URDF/USD where the pad is rigidly attached.
    # Matches the fixed child link added to the URDF for the tactile mount.
    link_name: str = "wrist_roll_tactile_pad_link"

    # Shape of the tactile map (rows, cols) — must match real-robot driver.
    shape: tuple[int, int] = (12, 32)

    # Center spans of the manual flat tactile patch in the 12-row and 32-col directions.
    # Defaults keep the width side at 28 mm and tile the 32-col side with the same sphere diameter.
    # (u_center_span, v_center_span). Will be centred on `pad_offset`.
    grid_size_m: tuple[float, float] = _DEFAULT_TACTILE_GRID_SIZE_M

    # Offset from the link origin to the pad centre, local frame (x, y, z) in m.
    # Calibrated from static tactile alignment in Isaac Sim.
    pad_offset: tuple[float, float, float] = (0.019, 0.0, -0.01072)

    # Rotation from the link frame to the pad frame as a unit quaternion (w, x, y, z).
    # Default applies the validated 90-degree local-Z alignment plus an extra 180-degree turn.
    pad_quat: tuple[float, float, float, float] = (0.7071067811865476, 0.0, 0.0, -0.7071067811865475)

    # Real-sensor normalization (same defaults as lerobot_tactile's TactileSensorConfig).
    # The sim produces raw ADC-like counts above baseline; the observation is then
    # normalized with the official flexitac code, exactly like the real driver.
    threshold: float = 25.0
    noise_scale: float = 30.0


@RobotConfig.register_subclass("isaacsim_so100_tactile_follower")
@dataclass
class IsaacSimSO100TactileFollowerConfig(RobotConfig):
    """Sim follower config — drop-in replacement for SO100TactileFollowerConfig."""

    # --- Asset paths ---
    # Pre-converted USD of the SO100 + tactile-pad arm. Default: the copy shipped in this package
    # (assets/gripper_so100_tactile/usd/so100_follower_tactile.usd).
    usd_path: str | None = None
    # Leave unset. Joint angles are then mapped 1:1 (leader degrees == URDF degrees), which is
    # the so101_new_calib convention (0 deg = middle of the calibrated range).
    # Setting a real-follower calibration JSON here linearly stretches that calibrated range onto
    # the URDF limits, which scales the angles by 0.93-0.995. It is kept only for backwards
    # compatibility.
    reference_calibration_fpath: str | None = None
    # Optional user-space wrist_roll zero offset in degrees. Useful when the
    # real follower's adjacent-to-gripper motor was left uncalibrated and the
    # leader/follower zero differs by a fixed ~90-degree bias.
    wrist_roll_user_offset_deg: float = -90.0

    # --- Scene ---
    # Table dimensions (x, y, z) in metres.  Default matches a small SO100 bench.
    table_size_m: tuple[float, float, float] = (0.8, 0.6, 0.72)
    # Arm base position in world frame (metres). Default: centred on table top.
    arm_base_pos_m: tuple[float, float, float] = (0.0, 0.0, 0.72)
    # Arm base orientation quaternion (w, x, y, z).
    arm_base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    enable_self_collisions: bool = False
    enable_ccd: bool = True
    enable_enhanced_determinism: bool = True
    solve_articulation_contact_last: bool = True
    arm_max_depenetration_velocity: float = 5.0
    arm_solver_position_iteration_count: int = 32
    arm_solver_velocity_iteration_count: int = 8
    table_contact_offset_m: float = 0.005
    table_rest_offset_m: float = 0.0
    arm_contact_offset_m: float = 0.005
    arm_rest_offset_m: float = 0.0
    peg_contact_offset_m: float = 0.003
    peg_rest_offset_m: float = 0.0

    # --- Peg / hole (task-specific) ---
    # Default task objects are generated directly in scene for data collection.
    # The position tuples below are the main values to tweak when you want to
    # move the peg or hole around on the tabletop.
    spawn_task_objects: bool = True
    peg_stl_path: str | None = None
    hole_stl_path: str | None = None
    peg_init_pos_m: tuple[float, float, float] = (0.26, -0.04, 0.7575)
    hole_init_pos_m: tuple[float, float, float] = (0.28, 0.04, 0.73)
    peg_diameter_m: float = 0.022
    peg_height_m: float = 0.055
    peg_mass_kg: float = 0.03
    peg_color_rgb: tuple[float, float, float] = (0.25, 0.55, 0.95)
    hole_diameter_m: float = 0.070
    hole_wall_thickness_m: float = 0.001
    hole_height_m: float = 0.040
    hole_color_rgb: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # --- Cameras ---
    # Same CameraConfig type as lerobot, but the sim follower treats each entry
    # as an IsaacSim camera spec (poses/intrinsics to be set via a helper).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # --- Tactile pad(s) ---
    # Named entries so the observation-key naming matches the real class:
    #   obs["observation.tactile.primary"] -> (12, 32) map
    tactile_pads: dict[str, SimTactilePadConfig] = field(
        default_factory=lambda: {"primary": SimTactilePadConfig()}
    )

    # --- Sim runtime ---
    device: str = "cuda:0"
    headless: bool = False
    # 60 Hz physics by default.
    sim_dt: float = 1.0 / 60.0
    # Control rate of the lerobot loop that drives this robot. It must equal `--fps` of
    # lerobot-teleoperate and `--dataset.fps` of lerobot-record. Each send_action advances the
    # sim by exactly 1/fps (1 / (fps * sim_dt) physics sub-steps), so sim time matches dataset time.
    fps: int = 30

    # --- Debug / visualisation ---
    # Print a leader-vs-sim joint table every N control steps (0 = off).
    debug_joint_map_every_n: int = 0
    # Print tactile stats (raw counts, active taxels, min distance) every N control steps (0 = off).
    debug_tactile_every_n: int = 0
    # Dock a 12x32 tactile heatmap window in the Isaac Sim GUI (ignored when headless).
    tactile_heatmap_panel: bool = True

    # --- Policy-parity knobs ---
    # Match the real SO follower default so leader->sim teleop uses the same body-joint
    # units (degrees) and gripper convention (0-100) unless explicitly overridden.
    use_degrees: bool = True
    # `max_relative_target` is a no-op in sim but kept for API parity.
    max_relative_target: float | dict[str, float] | None = None
    disable_torque_on_disconnect: bool = True

