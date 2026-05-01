"""IsaacSim-backed SO100 tactile follower — lerobot `Robot` subclass.

Observation / action key names are identical to SO100TactileFollower so a
dataset or policy trained on either robot is load-compatible with the other.

Observation dict keys:
    shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
    wrist_flex.pos, wrist_roll.pos, gripper.pos     -> float
    top                                              -> (H, W, 3) uint8
    observation.tactile.<pad_name>                   -> (12, 32) float32
Action dict keys:
    shoulder_pan.pos, ..., gripper.pos               -> float

Units follow the real SO follower when `config.use_degrees=True`: body joints
are exposed in degrees and `gripper.pos` uses the familiar 0-100 convention.
Internally, the sim still runs on articulation joint units.
"""

from __future__ import annotations

import logging
import math
from functools import cached_property
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.motors import MotorCalibration
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import OBS_TACTILE

from .isaacsim_follower_config import IsaacSimSO100TactileFollowerConfig

logger = logging.getLogger(__name__)

# Joint order matching lerobot's FeetechMotorsBus motor registration.
_SO100_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_SO100_BODY_JOINTS: tuple[str, ...] = _SO100_JOINTS[:-1]
_STS3215_RESOLUTION = 4095.0


class IsaacSimSO100TactileFollower(Robot):
    """Sim twin of SO100TactileFollower driven by IsaacLab."""

    config_class = IsaacSimSO100TactileFollowerConfig
    name = "isaacsim_so100_tactile_follower"

    def __init__(self, config: IsaacSimSO100TactileFollowerConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

        self._sim = None
        self._app = None
        self._arm = None
        self._cameras: dict[str, Any] = {}
        self._tactile_sensors: dict[str, Any] = {}
        self._tactile_target_trackers: dict[str, list[dict[str, Any]]] = {}

        # joint_name -> column index in arm.data.joint_pos tensor
        self._joint_idx: dict[str, int] = {}
        self._joint_limits: dict[str, tuple[float, float]] = {}
        self._user_joint_bounds: dict[str, tuple[float, float]] = {}
        self._joint_centers = None
        # pre-allocated target tensor (shape [1, n_joints])
        self._joint_target = None
        self._sim_dt: float = config.sim_dt

    # ------------------------------------------------------------------
    # Feature schemas
    # ------------------------------------------------------------------
    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {f"{j}.pos": float for j in _SO100_JOINTS}
        # cameras
        features["top"] = (480, 640, 3)
        # tactile pads
        for name, pad in self.config.tactile_pads.items():
            features[f"{OBS_TACTILE}.{name}"] = PolicyFeature(
                type=FeatureType.TACTILE, shape=pad.shape
            )
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in _SO100_JOINTS}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def _get_user_joint_offset(self, joint_name: str) -> float:
        if joint_name == "wrist_roll" and self.config.use_degrees:
            return float(self.config.wrist_roll_user_offset_deg)
        return 0.0

    def _load_reference_calibration(self) -> dict[str, MotorCalibration]:
        fpath = self.config.reference_calibration_fpath
        if not fpath:
            return {}

        calibration_path = Path(fpath).expanduser().resolve()
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Reference follower calibration file not found: {calibration_path}")

        with open(calibration_path) as f, draccus.config_type("json"):
            return draccus.load(dict[str, MotorCalibration], f)

    def _hardware_user_bounds_from_calibration(
        self, calibration: dict[str, MotorCalibration]
    ) -> dict[str, tuple[float, float]]:
        user_bounds: dict[str, tuple[float, float]] = {}
        for joint_name in _SO100_BODY_JOINTS:
            if joint_name not in calibration:
                continue
            cal = calibration[joint_name]
            if float(cal.range_min) <= 0.0 and float(cal.range_max) >= _STS3215_RESOLUTION:
                logger.warning(
                    "Ignoring reference calibration for %s because it spans the full encoder range "
                    "[%s, %s], which usually means that joint was not calibrated.",
                    joint_name,
                    cal.range_min,
                    cal.range_max,
                )
                continue
            mid = 0.5 * (float(cal.range_min) + float(cal.range_max))
            lower = (float(cal.range_min) - mid) * 360.0 / _STS3215_RESOLUTION
            upper = (float(cal.range_max) - mid) * 360.0 / _STS3215_RESOLUTION
            user_bounds[joint_name] = (lower, upper)

        if "gripper" in calibration:
            user_bounds["gripper"] = (0.0, 100.0)

        return user_bounds

    def _default_user_bounds(self) -> dict[str, tuple[float, float]]:
        user_bounds: dict[str, tuple[float, float]] = {}
        for joint_name, (lower, upper) in self._joint_limits.items():
            if joint_name == "gripper":
                user_bounds[joint_name] = (0.0, 100.0)
            elif self.config.use_degrees:
                user_bounds[joint_name] = (math.degrees(lower), math.degrees(upper))
            else:
                user_bounds[joint_name] = (lower, upper)
        return user_bounds

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return

        # Launch IsaacSim app
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(
            {"headless": self.config.headless, "device": self.config.device, "enable_cameras": True}
        )
        self._app = launcher.app

        # Build scene (ground + table + arm + cameras)
        from main_package.sim_tactile.scene.build_scene import build_scene
        self._sim, self._arm, self._cameras = build_scene(self.config)

        # Instantiate tactile sensors
        from main_package.sim_tactile.sensors.pad_on_wrist_roll import build_tactile_pad_sensor
        query_targets = ["/World/Table"]
        if getattr(self.config, "spawn_task_objects", False) or self.config.hole_stl_path:
            query_targets.append("/World/Hole")
        if getattr(self.config, "spawn_task_objects", False) or self.config.peg_stl_path:
            query_targets.append("/World/Peg")
        for pad_name, pad_cfg in self.config.tactile_pads.items():
            try:
                self._tactile_sensors[pad_name] = build_tactile_pad_sensor(
                    pad_cfg=pad_cfg,
                    arm_prim_path="/World/Arm",
                    query_target_prim_paths=query_targets,
                )
            except NotImplementedError:
                logger.warning(f"Tactile pad '{pad_name}' not yet implemented — returning zeros.")

        # Reset only after all sensors exist so IsaacLab can initialize them on PLAY.
        self._sim.reset()
        self._arm.reset()
        for cam in self._cameras.values():
            cam.reset()
        for sensor in self._tactile_sensors.values():
            sensor.reset()
        self._initialize_tactile_target_trackers()

        # Build joint index map only after reset/play initializes articulation buffers.
        joint_names = list(self._arm.data.joint_names)
        self._joint_idx = {n: i for i, n in enumerate(joint_names)}
        missing = [j for j in _SO100_JOINTS if j not in self._joint_idx]
        if missing:
            raise RuntimeError(f"URDF missing joints: {missing}. Found: {joint_names}")

        joint_limits = self._arm.data.joint_pos_limits[0]
        self._joint_limits = {
            name: (
                float(joint_limits[idx, 0].item()),
                float(joint_limits[idx, 1].item()),
            )
            for name, idx in self._joint_idx.items()
        }
        self._user_joint_bounds = self._default_user_bounds()

        reference_calibration = self._load_reference_calibration()
        if reference_calibration:
            self._user_joint_bounds.update(self._hardware_user_bounds_from_calibration(reference_calibration))

        self._joint_centers = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
        joint_vel_zeros = self._joint_centers.new_zeros(self._joint_centers.shape)
        center_batch = self._joint_centers.unsqueeze(0)
        self._arm.write_joint_state_to_sim(center_batch, joint_vel_zeros.unsqueeze(0))
        self._arm.set_joint_position_target(center_batch)
        self._arm.write_data_to_sim()
        self._sim.step()
        self._arm.update(self._sim_dt)

        # Pre-allocate joint target tensor once articulation buffers exist.
        self._joint_target = self._arm.data.joint_pos.clone()

        self.configure()
        self._connected = True
        if self._tactile_sensors:
            tactile_keys = [f"{OBS_TACTILE}.{name}" for name in self._tactile_sensors]
            logger.info(
                "Registered tactile pads %s with observation keys %s.",
                sorted(self._tactile_sensors),
                tactile_keys,
            )
        if reference_calibration:
            logger.info(
                "Loaded reference follower calibration from %s for user-space joint bounds.",
                Path(self.config.reference_calibration_fpath).expanduser().resolve(),
            )
        if self.config.use_degrees:
            logger.info("IsaacSim follower joint interface: body joints in degrees, gripper in 0-100.")
        logger.info(f"{self} connected (IsaacSim).")

    @property
    def is_calibrated(self) -> bool:
        return True  # no encoder calibration needed in sim

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @staticmethod
    def _rigidprim_get_world_pose_numpy(rigid_prim_obj: Any) -> tuple[np.ndarray, np.ndarray]:
        def _to_numpy_1d(value: Any, size: int) -> np.ndarray:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.size < size:
                raise ValueError(f"Expected at least {size} values, got {array.size}.")
            return array[:size]

        if hasattr(rigid_prim_obj, "get_world_pose"):
            pos, quat = rigid_prim_obj.get_world_pose()
        elif hasattr(rigid_prim_obj, "get_world_poses"):
            pos, quat = rigid_prim_obj.get_world_poses()
        else:
            raise AttributeError("RigidPrim has neither get_world_pose nor get_world_poses")
        return _to_numpy_1d(pos, 3), _to_numpy_1d(quat, 4)

    def _initialize_tactile_target_trackers(self) -> None:
        import torch
        from pxr import UsdPhysics

        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils

        try:
            from omni.isaac.core.prims import RigidPrim
        except ImportError:
            try:
                from isaacsim.core.prims import RigidPrim
            except ImportError:
                RigidPrim = None

        self._tactile_target_trackers.clear()
        stage = sim_utils.get_current_stage()
        if stage is None:
            logger.warning("Cannot initialize tactile target trackers because the USD stage is unavailable.")
            return

        def _construct_rigid_prim(rb_path: str):
            try:
                return RigidPrim(str(rb_path))
            except TypeError:
                pass
            try:
                return RigidPrim(str(rb_path), name=str(rb_path).replace("/", "_"))
            except TypeError:
                pass
            return RigidPrim(prim_path=str(rb_path))

        for pad_name, sensor in self._tactile_sensors.items():
            target_paths = list(getattr(sensor.cfg, "target_mesh_prim_paths", None) or [])
            if not target_paths:
                target_path = getattr(sensor.cfg, "target_mesh_prim_path", None)
                if target_path:
                    target_paths = [str(target_path)]

            trackers: list[dict[str, Any]] = []
            for target_index, query_path in enumerate(target_paths):
                tracker: dict[str, Any] = {
                    "target_index": int(target_index),
                    "query_path": str(query_path),
                    "mode": "stage",
                }

                prim = stage.GetPrimAtPath(str(query_path))
                if prim is None or not prim.IsValid():
                    logger.warning("Tactile target prim is invalid: %s", query_path)
                    trackers.append(tracker)
                    continue

                rigid_body_prim = None
                current = prim
                while current.IsValid() and not current.IsPseudoRoot():
                    if current.HasAPI(UsdPhysics.RigidBodyAPI) or current.HasAPI(UsdPhysics.MassAPI):
                        rigid_body_prim = current
                        break
                    current = current.GetParent()

                if rigid_body_prim is not None and RigidPrim is not None:
                    try:
                        rigid_prim = _construct_rigid_prim(rigid_body_prim.GetPath().pathString)
                        if hasattr(rigid_prim, "initialize"):
                            rigid_prim.initialize()

                        mesh_pos_w, mesh_quat_w = sim_utils.resolve_prim_pose(prim)
                        body_pos_w, body_quat_w = sim_utils.resolve_prim_pose(rigid_body_prim)

                        mesh_pos_t = torch.tensor(mesh_pos_w, dtype=torch.float32)
                        mesh_quat_t = torch.tensor(mesh_quat_w, dtype=torch.float32)
                        body_pos_t = torch.tensor(body_pos_w, dtype=torch.float32)
                        body_quat_t = torch.tensor(body_quat_w, dtype=torch.float32)

                        body_quat_inv_t = math_utils.quat_inv(body_quat_t)
                        tracker.update(
                            {
                                "mode": "rigid_prim",
                                "rigid_prim": rigid_prim,
                                "rigid_body_path": rigid_body_prim.GetPath().pathString,
                                "p_rel": math_utils.quat_apply(body_quat_inv_t, mesh_pos_t - body_pos_t),
                                "q_rel": math_utils.quat_mul(body_quat_inv_t, mesh_quat_t),
                            }
                        )
                        logger.info(
                            "Tactile target tracker: pad=%s target[%d]=%s via rigid body %s",
                            pad_name,
                            target_index,
                            query_path,
                            rigid_body_prim.GetPath().pathString,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Falling back to USD pose tracking for tactile target %s because RigidPrim init failed: %s",
                            query_path,
                            exc,
                        )
                else:
                    logger.info(
                        "Tactile target tracker: pad=%s target[%d]=%s via USD pose",
                        pad_name,
                        target_index,
                        query_path,
                    )

                trackers.append(tracker)

            self._tactile_target_trackers[pad_name] = trackers

    def _update_tactile_target_poses(self, pad_name: str, sensor: Any) -> None:
        import torch

        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils

        trackers = self._tactile_target_trackers.get(pad_name)
        if not trackers:
            return

        stage = sim_utils.get_current_stage()
        for tracker in trackers:
            target_index = int(tracker["target_index"])
            if tracker.get("mode") == "rigid_prim":
                try:
                    body_pos_np, body_quat_np = self._rigidprim_get_world_pose_numpy(tracker["rigid_prim"])
                    body_pos_t = torch.tensor(body_pos_np, dtype=torch.float32)
                    body_quat_t = torch.tensor(body_quat_np, dtype=torch.float32)
                    target_pos_t = body_pos_t + math_utils.quat_apply(body_quat_t, tracker["p_rel"])
                    target_quat_t = math_utils.quat_mul(body_quat_t, tracker["q_rel"])
                    sensor.set_target_pose(target_pos_t.numpy(), target_quat_t.numpy(), target_index=target_index)
                    continue
                except Exception:
                    pass

            if stage is None:
                continue

            prim = stage.GetPrimAtPath(str(tracker["query_path"]))
            if prim is None or not prim.IsValid():
                continue
            try:
                pos_w, quat_w = sim_utils.resolve_prim_pose(prim)
                sensor.set_target_pose(pos_w, quat_w, target_index=target_index)
            except Exception:
                continue

    def disconnect(self) -> None:
        if not self._connected and self._app is None and self._sim is None:
            return

        try:
            self._tactile_sensors.clear()
            self._tactile_target_trackers.clear()
            self._cameras.clear()
            self._joint_idx.clear()
            self._joint_limits.clear()
            self._user_joint_bounds.clear()
            self._joint_centers = None
            self._joint_target = None
            self._arm = None

            if self._sim is not None:
                try:
                    self._sim.clear_all_callbacks()
                except Exception:
                    pass
                try:
                    self._sim.clear_instance()
                except Exception:
                    pass
                self._sim = None
        finally:
            self._connected = False
            logger.info(f"{self} disconnected.")

    def close_app(self) -> None:
        if self._app is None:
            return
        try:
            self._app.close()
        finally:
            self._app = None

    def get_user_joint_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self._user_joint_bounds)

    def tactile_debug_snapshot(self) -> dict[str, dict[str, float | int | list[str] | list[float]]]:
        snapshot: dict[str, dict[str, float | int | list[str] | list[float]]] = {}
        for pad_name, sensor in self._tactile_sensors.items():
            tactile_points = sensor.data.tactile_points_w_per_sensor
            sdf_out = getattr(sensor, "_sdf_out", None)
            target_paths = list(getattr(sensor.cfg, "target_mesh_prim_paths", None) or [])
            patch_normal_w: list[float] | None = None

            try:
                import isaaclab.utils.math as math_utils
                import torch

                if sensor._pose_sensors and sensor._pose_sensors[0].is_initialized:
                    pose_sensor = sensor._pose_sensors[0]
                    body_quat_w = pose_sensor.data.quat_w[0, 0]
                    patch_quat_b = torch.tensor(
                        sensor._resolve_patch_offset_quat_list()[0],
                        device=body_quat_w.device,
                        dtype=torch.float32,
                    )
                    patch_quat_w = math_utils.quat_mul(body_quat_w.unsqueeze(0), patch_quat_b.unsqueeze(0)).squeeze(0)
                    patch_normal_w_t = math_utils.quat_apply(
                        patch_quat_w.unsqueeze(0),
                        torch.tensor([[0.0, 0.0, 1.0]], device=body_quat_w.device, dtype=torch.float32),
                    ).squeeze(0)
                    patch_normal_w = [float(v) for v in patch_normal_w_t.detach().cpu().tolist()]
            except Exception:
                patch_normal_w = None

            if tactile_points is None:
                snapshot[pad_name] = {
                    "fn_max": 0.0,
                    "fn_mean": 0.0,
                    "active_taxels": 0,
                    "sdf_min": float("inf"),
                    "sdf_mean": float("inf"),
                    "target_paths": target_paths,
                    "patch_normal_w": patch_normal_w or [],
                }
                continue

            fn = tactile_points[0, 0, :, 3].detach().float().cpu().numpy()
            if sdf_out is not None:
                sdf = sdf_out[0, 0].detach().float().cpu().numpy()
                sdf_min = float(np.min(sdf))
                sdf_mean = float(np.mean(sdf))
            else:
                sdf_min = float("inf")
                sdf_mean = float("inf")

            snapshot[pad_name] = {
                "fn_max": float(np.max(fn)),
                "fn_mean": float(np.mean(fn)),
                "active_taxels": int(np.count_nonzero(fn > 1.0e-6)),
                "sdf_min": sdf_min,
                "sdf_mean": sdf_mean,
                "target_paths": target_paths,
                "patch_normal_w": patch_normal_w or [],
            }
        return snapshot

    # ------------------------------------------------------------------
    # Per-step I/O
    # ------------------------------------------------------------------
    def _sim_to_user_linear(self, joint_name: str, joint_value: float) -> float:
        sim_lower, sim_upper = self._joint_limits[joint_name]
        user_lower, user_upper = self._user_joint_bounds[joint_name]
        sim_span = sim_upper - sim_lower
        if abs(sim_span) <= 1.0e-8:
            return float(user_lower + self._get_user_joint_offset(joint_name))
        alpha = (float(joint_value) - sim_lower) / sim_span
        return user_lower + alpha * (user_upper - user_lower) + self._get_user_joint_offset(joint_name)

    def _user_to_sim_linear(self, joint_name: str, joint_value: float) -> float:
        sim_lower, sim_upper = self._joint_limits[joint_name]
        user_lower, user_upper = self._user_joint_bounds[joint_name]
        user_value = float(joint_value) - self._get_user_joint_offset(joint_name)
        user_span = user_upper - user_lower
        if abs(user_span) <= 1.0e-8:
            return float(sim_lower)
        alpha = (user_value - user_lower) / user_span
        return sim_lower + alpha * (sim_upper - sim_lower)

    def _sim_to_user_joint_value(self, joint_name: str, joint_value: float) -> float:
        return self._sim_to_user_linear(joint_name, joint_value)

    def _user_to_sim_joint_value(self, joint_name: str, joint_value: float) -> float:
        return self._user_to_sim_linear(joint_name, joint_value)

    def _clamp_user_joint_value(self, joint_name: str, joint_value: float) -> float:
        bounds = self.get_user_joint_bounds().get(joint_name)
        if bounds is None:
            return float(joint_value)
        lower, upper = bounds
        return float(np.clip(joint_value, min(lower, upper), max(lower, upper)))

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise RuntimeError(f"{self} not connected")

        import torch
        obs: dict[str, Any] = {}

        # --- Joint positions (radians) ---
        joint_pos = self._arm.data.joint_pos[0]  # shape [n_joints]
        for joint in _SO100_JOINTS:
            idx = self._joint_idx[joint]
            obs[f"{joint}.pos"] = self._sim_to_user_joint_value(joint, float(joint_pos[idx].item()))

        # --- Cameras (H, W, 3) uint8 ---
        for cam_name, cam in self._cameras.items():
            cam.update(dt=self._sim_dt)
            rgb = cam.data.output.get("rgb")
            if rgb is not None:
                # shape: [1, H, W, 3] or [H, W, 3]
                arr = rgb[0].cpu().numpy() if rgb.ndim == 4 else rgb.cpu().numpy()
                # convert float [0,1] → uint8 if needed
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                obs[cam_name] = arr
            else:
                obs[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)

        # --- Tactile (12, 32) float32 ---
        for pad_name, pad_cfg in self.config.tactile_pads.items():
            obs_key = f"{OBS_TACTILE}.{pad_name}"
            sensor = self._tactile_sensors.get(pad_name)
            if sensor is not None:
                try:
                    self._update_tactile_target_poses(pad_name, sensor)
                    sensor.update(dt=self._sim_dt)
                    tactile_points = sensor.data.tactile_points_w_per_sensor
                    if tactile_points is None:
                        obs[obs_key] = np.zeros(pad_cfg.shape, dtype=np.float32)
                    else:
                        fn = tactile_points[0, 0, :, 3].detach().float().cpu().numpy()
                        obs[obs_key] = fn.reshape(pad_cfg.shape).astype(np.float32)
                except Exception as e:
                    logger.warning(f"Tactile read failed for '{pad_name}': {e}")
                    obs[obs_key] = np.zeros(pad_cfg.shape, dtype=np.float32)
            else:
                obs[obs_key] = np.zeros(pad_cfg.shape, dtype=np.float32)

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected:
            raise RuntimeError(f"{self} not connected")

        import torch

        # Write goal positions into target tensor
        for joint in _SO100_JOINTS:
            key = f"{joint}.pos"
            if key in action:
                idx = self._joint_idx[joint]
                clamped_joint_value = self._clamp_user_joint_value(joint, float(action[key]))
                self._joint_target[0, idx] = self._user_to_sim_joint_value(joint, clamped_joint_value)

        # Step sim `decimation` times
        for _ in range(self.config.decimation):
            self._arm.set_joint_position_target(self._joint_target)
            self._arm.write_data_to_sim()
            self._sim.step()
            self._arm.update(self._sim_dt)

        return {
            f"{j}.pos": self._sim_to_user_joint_value(j, float(self._joint_target[0, self._joint_idx[j]].item()))
            for j in _SO100_JOINTS
        }

    # ------------------------------------------------------------------
    # Dataset metadata (mirrors SO100TactileFollower)
    # ------------------------------------------------------------------
    def tactile_sensor_metadata(self) -> dict[str, dict]:
        return {
            name: {
                "shape": list(pad.shape),
                "link_name": pad.link_name,
                "grid_size_m": list(pad.grid_size_m),
                "pad_offset": list(pad.pad_offset),
                "source": "isaacsim_warp_sdf",
            }
            for name, pad in self.config.tactile_pads.items()
        }
