"""Config of the Isaac Sim SO100 tactile follower (`--robot.type=isaacsim_so100_tactile_follower`).

Sim counterpart of lerobot_tactile's `SOTactileFollowerConfig`: the serial/hardware fields are
replaced by scene, asset and sim-timing fields.
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
    """One tactile pad rigidly attached to a URDF link (12x32, like the real FlexiTac driver)."""

    link_name: str = "wrist_roll_tactile_pad_link"
    shape: tuple[int, int] = (12, 32)
    # Taxel-centre spans (rows, cols); 28 mm across the 12 rows, same pitch along the 32 cols.
    grid_size_m: tuple[float, float] = _DEFAULT_TACTILE_GRID_SIZE_M
    # Pad pose in the link frame (calibrated visually in Isaac Sim); quaternion is (w, x, y, z).
    pad_offset: tuple[float, float, float] = (0.019, 0.0, -0.01072)
    pad_quat: tuple[float, float, float, float] = (0.7071067811865476, 0.0, 0.0, -0.7071067811865475)
    # flexitac normalization parameters (lerobot_tactile TactileSensorConfig defaults).
    threshold: float = 25.0
    noise_scale: float = 30.0


@RobotConfig.register_subclass("isaacsim_so100_tactile_follower")
@dataclass
class IsaacSimSO100TactileFollowerConfig(RobotConfig):
    """Sim follower config; observation/action schema matches the real so_tactile_follower."""

    # --- Assets / joint mapping ---
    usd_path: str | None = None  # default: the USD shipped in assets/gripper_so100_tactile/usd/
    # Leave unset for the 1:1 so101_new_calib mapping; a calibration JSON rescales joint ranges.
    reference_calibration_fpath: str | None = None
    wrist_roll_user_offset_deg: float = -90.0

    # --- Scene ---
    table_size_m: tuple[float, float, float] = (0.8, 0.6, 0.72)
    arm_base_pos_m: tuple[float, float, float] = (0.0, 0.0, 0.72)
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

    # --- Task objects ---
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

    # --- Sensors ---
    cameras: dict[str, CameraConfig] = field(default_factory=dict)  # API parity; the scene builds "top"
    tactile_pads: dict[str, SimTactilePadConfig] = field(
        default_factory=lambda: {"primary": SimTactilePadConfig()}
    )

    # --- Sim runtime ---
    device: str = "cuda:0"
    headless: bool = False
    sim_dt: float = 1.0 / 60.0
    # Control rate; must equal the script's --fps / --dataset.fps. Each step advances the sim by 1/fps.
    fps: int = 30

    # --- Debug ---
    debug_joint_map_every_n: int = 0  # print a leader/target/sim joint table every N steps (0 = off)
    debug_tactile_every_n: int = 0  # print tactile stats every N steps (0 = off)
    tactile_heatmap_panel: bool = True  # docked 12x32 heatmap in the GUI

    # --- Parity with the real follower ---
    use_degrees: bool = True
    max_relative_target: float | dict[str, float] | None = None  # no-op in sim
    disable_torque_on_disconnect: bool = True
