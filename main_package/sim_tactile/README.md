# sim_tactile — IsaacSim port of lerobot_tactile (new-sync)

IsaacSim backend for the tactile-enabled SO100 workflow from
[`TNA001-AI/lerobot_tactile@new-sync`](https://github.com/TNA001-AI/lerobot_tactile/tree/new-sync).

The sim follower is implemented as a lerobot `Robot` subclass so the
existing record / train / eval scripts in lerobot_tactile work unchanged
once we point them at `--robot.type=isaacsim_so100_tactile_follower`.

## Status

Skeleton committed. TODO markers flag the implementation gaps:

| Area | File | Status |
|---|---|---|
| Lerobot Robot interface | `follower/isaacsim_follower.py` | skeleton |
| Config (draccus-registered) | `follower/isaacsim_follower_config.py` | ✅ |
| Scene builder (table + fixed arm) | `scene/build_scene.py` | skeleton |
| Warp SDF pad on Wrist_Roll_08c | `sensors/pad_on_wrist_roll.py` | skeleton |
| Teleop-record entrypoint | `scripts/teleop_record.py` | ✅ lerobot_dev wrapper |
| Sim policy eval entrypoint | `scripts/eval_policy_sim.py` | skeleton |
| Warp SDF tactile sensor (IsaacLab) | `source/isaaclab/isaaclab/sensors/warp_sdf_tactile/` | ✅ in main tree |
| Lerobot vendor clone | `_vendor/lerobot_tactile/` | ✅ new-sync branch |
| Onshape-to-robot config | `assets/onshape_export/config.json` | ✅ (user adds API key) |
| 16 mm peg / hole STL | `assets/peg_and_hole/` | ✅ |

## Observation / action schema

Matches real `SO100TactileFollower` so a dataset / policy trained on sim
data is load-compatible with the real robot (and vice versa).

Observation dict (from `robot.get_observation()`):

| key | type | shape |
|---|---|---|
| `shoulder_pan.pos` | float | scalar |
| `shoulder_lift.pos` | float | scalar |
| `elbow_flex.pos` | float | scalar |
| `wrist_flex.pos` | float | scalar |
| `wrist_roll.pos` | float | scalar |
| `gripper.pos` | float | scalar |
| `<cam_name>` | uint8 | `(H, W, 3)` |
| `observation.tactile.<pad_name>` | float32 | `(12, 32)` |

Action dict: the six `*.pos` keys. Units = RANGE_M100_100 by default,
degrees if `config.use_degrees=True` — same as the real class.

## First-time setup

```bash
# 1. Onshape API credentials (do NOT paste secrets in chat)
export ONSHAPE_API_KEY=xxxx
export ONSHAPE_API_SECRET=yyyy

# 2. Export the gripper URDF
cd main_package/sim_tactile/assets/onshape_export
onshape-to-robot .
mkdir -p ../gripper_so100_tactile/{urdf,meshes}
mv robot.urdf ../gripper_so100_tactile/urdf/so100_follower_tactile.urdf
mv *.stl ../gripper_so100_tactile/meshes/
cd -

# 3. Install the vendored lerobot (editable)
pip install -e main_package/sim_tactile/_vendor/lerobot_tactile

# 4. Point PYTHONPATH at the repo root so `import main_package.sim_tactile…` resolves.
#    isaaclab.sh already does this; for bare python:
export PYTHONPATH=/home/fan/workspace/IsaacLab:$PYTHONPATH
```

## Recommended implementation order

1. **Fill `scene/build_scene.py`** — table, fixed-base arm, dome light, one
   wrist camera. Run an empty-loop demo (no leader, no tactile) to confirm
   the URDF loads and joints move.
2. **Fill `IsaacSimSO100TactileFollower.connect` / `send_action` / `get_observation`** —
   wire to the scene, stub `tactile` as zeros. Write a 20-line test
   script that drives joint sinusoids and prints the obs dict.
3. **Teleop**: run `scripts/teleop_record.py` against the real SO100 leader.
   No tactile yet. Goal: an episode parquet that the lerobot viewer opens.
4. **Tactile**: fill `sensors/pad_on_wrist_roll.py`, attach one warp SDF
   sensor to `Wrist_Roll_08c`. Verify by closing the gripper onto a
   surface and watching the (12, 32) map activate.
5. **Task**: drop in the peg/hole (user provides peg when ready). Collect
   ≈50 demonstration episodes.
6. **Train + eval**: run the lerobot `train.py` on the dataset with / without
   tactile features; run `scripts/eval_policy_sim.py` on the checkpoints.

## Do / Don't

- Do reuse lerobot scripts as-is — swap the robot via CLI, not by forking.
- Do keep observation / action key names identical to real
  `SO100TactileFollower`.
- Don't touch `main_package/bridge/` — that's an unrelated project.
- Don't floating-base the arm; its root stays fixed on the table.
