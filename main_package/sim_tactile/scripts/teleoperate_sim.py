"""Real SO100 leader  →  IsaacSim SO100-tactile follower, live teleoperation.

This wrapper mirrors `teleop_record.py` but routes to lerobot_dev's
`lerobot_teleoperate.py`, which is useful for validating the hardware leader
to sim-follower control loop before recording datasets.

Example:

    ./isaaclab.sh -p main_package/sim_tactile/scripts/teleoperate_sim.py \
        --robot.type=isaacsim_so100_tactile_follower \
        --robot.urdf_path=main_package/sim_tactile/assets/gripper_so100_tactile/urdf/so100_follower_tactile.urdf \
        --teleop.type=so100_leader \
        --teleop.port=/dev/ttyACM0 \
        --display_data=true
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from main_package.sim_tactile.scripts.teleop_record import _bootstrap_lerobot_dev

_SO100_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class _DebugOptions:
    joint_map: bool = False
    joint_map_every_n: int = 15
    limit_monitor: bool = False
    limit_margin_ratio: float = 0.08
    tactile_monitor: bool = False
    tactile_monitor_every_n: int = 15


def _consume_debug_cli_args() -> _DebugOptions:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--debug-joint-map", action="store_true")
    parser.add_argument("--debug-joint-map-every-n", type=int, default=15)
    parser.add_argument("--debug-limit-monitor", action="store_true")
    parser.add_argument("--debug-limit-margin-ratio", type=float, default=0.08)
    parser.add_argument("--debug-tactile-monitor", action="store_true")
    parser.add_argument("--debug-tactile-monitor-every-n", type=int, default=15)
    known_args, remaining_args = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining_args]
    return _DebugOptions(
        joint_map=bool(known_args.debug_joint_map),
        joint_map_every_n=max(int(known_args.debug_joint_map_every_n), 1),
        limit_monitor=bool(known_args.debug_limit_monitor),
        limit_margin_ratio=float(known_args.debug_limit_margin_ratio),
        tactile_monitor=bool(known_args.debug_tactile_monitor),
        tactile_monitor_every_n=max(int(known_args.debug_tactile_monitor_every_n), 1),
    )


def _joint_limit_margin_ratio(value: float, bounds: tuple[float, float] | None) -> float | None:
    if bounds is None:
        return None
    lower, upper = bounds
    span = upper - lower
    if abs(span) <= 1.0e-8:
        return None
    lower_margin = (value - lower) / span
    upper_margin = (upper - value) / span
    return float(min(lower_margin, upper_margin))


def _format_joint_debug_snapshot(
    *,
    step_idx: int,
    leader_action: dict[str, float],
    sent_robot_action: dict[str, float],
    robot_observation: dict[str, float],
    joint_bounds: dict[str, tuple[float, float]],
) -> str:
    lines = [
        f"[teleoperate_sim][joint-map] step={step_idx}",
        "joint              leader    target       sim      err     ctr   lim%",
    ]
    for joint_name in _SO100_JOINTS:
        key = f"{joint_name}.pos"
        leader_value = float(leader_action.get(key, float("nan")))
        target_value = float(sent_robot_action.get(key, float("nan")))
        sim_value = float(robot_observation.get(key, float("nan")))
        bounds = joint_bounds.get(joint_name)
        center_value = 0.5 * (bounds[0] + bounds[1]) if bounds is not None else float("nan")
        limit_margin = _joint_limit_margin_ratio(sim_value, bounds)
        lines.append(
            f"{joint_name:<14} {leader_value:8.2f} {target_value:9.2f} {sim_value:9.2f}"
            f" {sim_value - target_value:8.2f} {sim_value - center_value:7.2f}"
            f" {100.0 * limit_margin:6.1f}" if limit_margin is not None else
            f"{joint_name:<14} {leader_value:8.2f} {target_value:9.2f} {sim_value:9.2f}"
            f" {sim_value - target_value:8.2f} {sim_value - center_value:7.2f} {'n/a':>6}"
        )
    return "\n".join(lines)


def _format_limit_warning(
    *,
    robot_observation: dict[str, float],
    joint_bounds: dict[str, tuple[float, float]],
    margin_ratio_threshold: float,
) -> str | None:
    near_limit_items: list[str] = []
    for joint_name in _SO100_JOINTS:
        key = f"{joint_name}.pos"
        if key not in robot_observation:
            continue
        margin_ratio = _joint_limit_margin_ratio(float(robot_observation[key]), joint_bounds.get(joint_name))
        if margin_ratio is None:
            continue
        if margin_ratio <= margin_ratio_threshold:
            near_limit_items.append(f"{joint_name}:{100.0 * margin_ratio:.1f}%")
    if not near_limit_items:
        return None
    return "[teleoperate_sim][limit] near joint limits -> " + ", ".join(near_limit_items)


def _format_tactile_debug_snapshot(
    *,
    step_idx: int,
    tactile_snapshot: dict[str, dict[str, float | int | list[str] | list[float]]],
) -> str:
    lines = [f"[teleoperate_sim][tactile] step={step_idx}"]
    for pad_name, stats in tactile_snapshot.items():
        lines.append(
            "  "
            f"pad={pad_name} "
            f"fn_max={float(stats['fn_max']):.6f} "
            f"fn_mean={float(stats['fn_mean']):.6f} "
            f"active={int(stats['active_taxels'])} "
            f"sdf_min={float(stats['sdf_min']):.6f} "
            f"sdf_mean={float(stats['sdf_mean']):.6f}"
        )
        patch_normal_w = list(stats.get("patch_normal_w", []))
        if patch_normal_w:
            lines.append(
                "    "
                f"patch_normal_w=({float(patch_normal_w[0]):+.4f}, "
                f"{float(patch_normal_w[1]):+.4f}, "
                f"{float(patch_normal_w[2]):+.4f})"
            )
        target_paths = list(stats.get("target_paths", []))
        if target_paths:
            lines.append(f"    targets={target_paths}")
    return "\n".join(lines)


def _teleop_loop_with_debug(
    *,
    teleop,
    robot,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    display_data: bool,
    duration: float | None,
    display_compressed_images: bool,
    debug_options: _DebugOptions,
) -> None:
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import log_rerun_data

    start = time.perf_counter()
    step_idx = 0
    joint_bounds_getter = getattr(robot, "get_user_joint_bounds", None)
    joint_bounds = joint_bounds_getter() if callable(joint_bounds_getter) else {}
    tactile_snapshot_getter = getattr(robot, "tactile_debug_snapshot", None)

    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()
        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        sent_robot_action = robot.send_action(robot_action_to_send)

        need_post_obs = debug_options.limit_monitor or (debug_options.joint_map and step_idx % debug_options.joint_map_every_n == 0)
        post_obs = robot.get_observation() if need_post_obs else None

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

        if debug_options.joint_map and step_idx % debug_options.joint_map_every_n == 0 and post_obs is not None:
            print(
                _format_joint_debug_snapshot(
                    step_idx=step_idx,
                    leader_action=raw_action,
                    sent_robot_action=sent_robot_action,
                    robot_observation=post_obs,
                    joint_bounds=joint_bounds,
                ),
                flush=True,
            )

        if debug_options.limit_monitor and post_obs is not None:
            warning = _format_limit_warning(
                robot_observation=post_obs,
                joint_bounds=joint_bounds,
                margin_ratio_threshold=debug_options.limit_margin_ratio,
            )
            if warning is not None:
                print(warning, flush=True)

        if (
            debug_options.tactile_monitor
            and callable(tactile_snapshot_getter)
            and step_idx % debug_options.tactile_monitor_every_n == 0
        ):
            print(
                _format_tactile_debug_snapshot(
                    step_idx=step_idx,
                    tactile_snapshot=tactile_snapshot_getter(),
                ),
                flush=True,
            )

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))

        if duration is not None and time.perf_counter() - start >= duration:
            return

        step_idx += 1


def _main() -> None:
    debug_options = _consume_debug_cli_args()
    _bootstrap_lerobot_dev()

    # Importing the follower package has the side-effect of registering the
    # IsaacSim follower subclass with RobotConfig's draccus registry.
    import main_package.sim_tactile.follower  # noqa: F401

    import rerun as rr

    from lerobot.configs import parser
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleop_loop
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.constants import OBS_TACTILE
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import init_rerun
    from main_package.sim_tactile.visualization.tactile_heatmap_panel import create_tactile_heatmap_panel

    def _teleoperate(cfg) -> None:
        init_logging()
        logging.info(pformat(asdict(cfg)))
        if cfg.display_data:
            init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
        display_compressed_images = (
            True
            if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
            else cfg.display_compressed_images
        )

        teleop = make_teleoperator_from_config(cfg.teleop)
        robot = make_robot_from_config(cfg.robot)
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        teleop.connect()
        robot.connect()

        tactile_panel = None
        tactile_obs_key = None
        tactile_pad_cfg = None
        robot_cfg = getattr(robot, "config", None)
        if getattr(robot_cfg, "headless", True) is False:
            tactile_pads = getattr(robot_cfg, "tactile_pads", None) or {}
            if tactile_pads:
                tactile_name, tactile_pad_cfg = next(iter(tactile_pads.items()))
                tactile_obs_key = f"{OBS_TACTILE}.{tactile_name}"
                tactile_panel = create_tactile_heatmap_panel(
                    num_rows=int(tactile_pad_cfg.shape[0]),
                    num_cols=int(tactile_pad_cfg.shape[1]),
                    title=f"SO100 Tactile Heatmap ({tactile_name})",
                )

        if tactile_panel is not None and tactile_obs_key is not None:
            base_get_observation = robot.get_observation

            def _get_observation_with_tactile_panel():
                obs = base_get_observation()
                tactile_grid = obs.get(tactile_obs_key)
                if tactile_grid is not None:
                    tactile_panel["update"](np.asarray(tactile_grid, dtype=np.float32))
                return obs

            robot.get_observation = _get_observation_with_tactile_panel  # type: ignore[method-assign]

        try:
            if debug_options.joint_map or debug_options.limit_monitor:
                _teleop_loop_with_debug(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    display_compressed_images=display_compressed_images,
                    debug_options=debug_options,
                )
            else:
                teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    display_compressed_images=display_compressed_images,
                )
        except KeyboardInterrupt:
            pass
        finally:
            if cfg.display_data:
                rr.rerun_shutdown()
            print("[teleoperate_sim] disconnecting robot...", flush=True)
            try:
                robot.disconnect()
            finally:
                print("[teleoperate_sim] disconnecting teleop...", flush=True)
                try:
                    teleop.disconnect()
                finally:
                    close_app = getattr(robot, "close_app", None)
                    if callable(close_app):
                        print("[teleoperate_sim] closing simulation app...", flush=True)
                        close_app()

    _teleoperate.__annotations__["cfg"] = TeleoperateConfig
    teleoperate = parser.wrap()(_teleoperate)

    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    _main()