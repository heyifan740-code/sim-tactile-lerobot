# Vendored: WarpSdfTactileSensor

- Source: https://github.com/binghao-huang/FlexiTac-IsaacSim-Simulation
- Path: `source/isaaclab/isaaclab/sensors/warp_sdf_tactile/`
- Commit: `dc2e1c6607a1b1a0e4b196bb4e2d4ccdc1afdcb0`

The files are copied as they are, except for one change: the relative imports are rewritten as absolute ones, so the package can live outside `isaaclab.sensors`.

- `warp_sdf_tactile_sensor.py`: `from ..contact_sensor` → `from isaaclab.sensors.contact_sensor`, `from ..sensor_base` → `from isaaclab.sensors.sensor_base`
- `warp_sdf_tactile_cfg.py`: `from ..sensor_base_cfg` → `from isaaclab.sensors.sensor_base_cfg`

Do not edit these files. Adapt the calling code in `../pad_on_wrist_roll.py` instead.
