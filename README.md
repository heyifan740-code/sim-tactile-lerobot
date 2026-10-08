# sim-tactile-lerobot

Run the [LeFlexiTac](https://tna001-ai.github.io/LeFlexiTac/) tactile workflow in **Isaac Sim**.
It ships as a **LeRobot robot plugin**: keep using the official lerobot commands (`lerobot-calibrate`, `lerobot-teleoperate`, `lerobot-record`, `lerobot-train`). Where the real setup says `--robot.type=so_tactile_follower`, pass `--robot.type=isaacsim_so100_tactile_follower` instead.

A physical SO100/SO101 **leader arm** drives a simulated SO100 follower. The follower carries a 12×32 FlexiTac tactile pad on the fixed jaw.
The recorded datasets have the same schema as the real-robot datasets: `action`, `observation.state`, `observation.images.top` (480×640) and `observation.tactile.primary` (12×32, float32). Policies train with the unmodified lerobot training code.

Upstreams (single sources of truth):

| Component | Upstream | How it is used |
|---|---|---|
| Real-robot stack (teleop / record / train / eval, tactile policies) | [TNA001-AI/lerobot_tactile](https://github.com/TNA001-AI/lerobot_tactile) `main` | git submodule `third_party/lerobot_tactile`, plus a syntax-only Python 3.11 patch |
| Tactile sensor simulation | [binghao-huang/FlexiTac-IsaacSim-Simulation](https://github.com/binghao-huang/FlexiTac-IsaacSim-Simulation) `WarpSdfTactileSensor` | vendored unmodified in `src/lerobot_robot_isaacsim_tactile/warp_sdf_tactile/` (see its `UPSTREAM.md`) |
| Tactile normalization | [PyFlexiTac](https://github.com/WT-MM/PyFlexiTac) (`flexitac`) | the real driver's own `_normalize`, applied to the simulated readings |

```
sim-tactile-lerobot/
├── src/lerobot_robot_isaacsim_tactile/   # the plugin (pip install -e .)
│   ├── config_isaacsim_so100_tactile_follower.py   # --robot.* options
│   ├── isaacsim_so100_tactile_follower.py          # lerobot Robot: connect / get_observation / send_action
│   ├── scene.py                                    # table + fixed-base arm + peg/hole + top camera
│   ├── tactile_pad.py                              # 12×32 pad on the fixed jaw, one sensor per target
│   ├── warp_sdf_tactile/                           # upstream FlexiTac sensor (unmodified)
│   └── assets/gripper_so100_tactile/               # URDF, meshes, converted USD
├── third_party/lerobot_tactile/          # submodule: official lerobot_tactile main
├── patches/                              # lerobot_tactile Python 3.11 patch + apply script
├── requirements-isaacsim.txt             # lerobot runtime deps verified with Isaac Sim 5.1
├── tools/                                # dev demos (static pad view, analytic-box tactile, joint sinusoid)
└── assets/                               # CAD sources (Onshape export config, peg/hole STLs)
```

---

## Installation

Use one Python environment that has Isaac Sim and IsaacLab installed (**Isaac Sim 5.1, IsaacLab 2.3, Python 3.11**). See the [IsaacLab pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

```bash
conda activate isaaclab                     # your Isaac Sim 5.1 / IsaacLab env

git clone --recurse-submodules https://github.com/heyifan740-code/sim-tactile-lerobot.git
cd sim-tactile-lerobot

# 1. lerobot_tactile (official main). It declares Python >= 3.12, but Isaac Sim 5.1 is 3.11:
#    apply the syntax-only patch (4 files, no logic change), then install without deps so
#    pip does not replace Isaac Sim's numpy / torch.
bash patches/apply_lerobot_py311.sh
pip install -r requirements-isaacsim.txt
pip install --no-deps --ignore-requires-python -e third_party/lerobot_tactile

# 2. this plugin
pip install --no-deps -e .

# check: the official CLI sees the sim robot
lerobot-record --robot.type=isaacsim_so100_tactile_follower --help | grep -- --robot.fps
```

The leader's serial port must be readable: `sudo usermod -aG dialout $USER`, then log out and back in.

---

## Workflow (the same commands as real-robot lerobot_tactile)

The sim follower runs at `--robot.fps=30` by default. **Pass the same rate to the script**: `--fps=30` for teleoperate and `--dataset.fps=30` for record (the record default). Each control step then advances the sim by exactly 1/30 s, so sim time equals dataset time. The robot warns if the loop rate does not match.

### 0. Calibrate the leader (official, once)

```bash
lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_leader
```

Joint angles are mapped 1:1 onto the URDF (`so101_new_calib` convention: 0° = middle of the calibrated range).
If the sim pose drifts from the leader pose, recalibrate the leader. Do not add an offset in the sim.

### 1. Teleoperate

```bash
lerobot-teleoperate \
  --robot.type=isaacsim_so100_tactile_follower \
  --teleop.type=so100_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_leader \
  --fps=30 --display_data=true
```

The Isaac Sim GUI opens with a docked 12×32 tactile heatmap. `--display_data=true` shows the camera and tactile streams in rerun, as on the real robot.

### 2. Record a dataset

```bash
lerobot-record \
  --robot.type=isaacsim_so100_tactile_follower \
  --teleop.type=so100_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_leader \
  --dataset.repo_id=<hf_user>/sim_so100_tactile_peg \
  --dataset.single_task="peg insertion" \
  --dataset.num_episodes=50 --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false --display_data=true
```

Keyboard controls are the same as the real robot: → ends the episode, ← re-records it, Esc stops.

### 3. Train (with / without tactile)

Training needs no simulator. It runs in any lerobot_tactile environment, including the official Python 3.12 one:

```bash
# with tactile
lerobot-train --dataset.repo_id=<hf_user>/sim_so100_tactile_peg --policy.type=act \
  --policy.use_tactile=true --policy.tactile_features='["observation.tactile.primary"]' \
  --policy.device=cuda --policy.push_to_hub=false --output_dir=outputs/train/act_sim_tactile

# ablation: without tactile
lerobot-train --dataset.repo_id=<hf_user>/sim_so100_tactile_peg --policy.type=act \
  --policy.use_tactile=false \
  --policy.device=cuda --policy.push_to_hub=false --output_dir=outputs/train/act_sim_notactile
```

### 4. Evaluate in sim

Same as on the real robot: roll the policy out with `lerobot-record --policy.path`.

```bash
lerobot-record \
  --robot.type=isaacsim_so100_tactile_follower \
  --policy.path=outputs/train/act_sim_tactile/checkpoints/last/pretrained_model \
  --dataset.repo_id=<hf_user>/eval_sim_so100_tactile_peg \
  --dataset.single_task="peg insertion" --dataset.num_episodes=10 --dataset.episode_time_s=30 \
  --dataset.push_to_hub=false
```

The sim is lock-stepped: each frame advances the physics by exactly 1/fps. A slow policy loop therefore only slows down wall-clock time and never changes the dynamics.

---

## Robot options (`--robot.*`)

| Option | Default | Meaning |
|---|---|---|
| `fps` | `30` | Control rate; must equal the script's `--fps` / `--dataset.fps` |
| `headless` | `false` | Run Isaac Sim without GUI (the camera still renders) |
| `debug_joint_map_every_n` | `0` | Print a leader / target / sim joint table every N steps |
| `debug_tactile_every_n` | `0` | Print tactile stats (raw counts, active taxels, min distance) every N steps |
| `tactile_heatmap_panel` | `true` | Docked 12×32 heatmap in the GUI |
| `enable_self_collisions` | `false` | Arm self-collision |
| `spawn_task_objects`, `peg_*`, `hole_*` | peg Ø22 mm, ring hole | Task objects on the table (`peg_stl_path` / `hole_stl_path` to use meshes) |
| `sim_dt` | `1/60` | Physics step; `1/(fps·sim_dt)` must be an integer |

---

## Tactile model

- **Geometry:** a 12×32 taxel grid with a 28 mm / 12 = 2.33 mm pitch, 28 × 74.7 mm in total, on `wrist_roll_tactile_pad_link`. Each taxel is a sphere of radius 1.17 mm (half the pitch) resting on the inner face of the fixed jaw. The distance is measured from the sphere centre.
- **Distance:** the upstream `WarpSdfTactileSensor` computes the **signed** (winding-number) distance *sdf* from each taxel to the Table, Hole and Peg meshes: positive outside, negative inside. It handles one mesh per sensor, so the plugin builds one sensor per target and keeps the closest one. Target meshes must be watertight (true for the built-in peg, hole and table).
- **Raw counts:** `clamp(K · (r − sdf), 0, 255)`, where r is the taxel radius. The signal starts when an object touches a sphere, reaches 125 counts (threshold 25 plus full scale 100 of the real sensor) when the object has pressed the sphere all the way to the jaw face, and keeps growing beyond that. It is continuous and monotonic, so deep contact never loses signal.
- **Jaw collider:** the arm USD uses **convex decomposition**, with shrink-wrap on the jaw links, so objects rest on the real jaw face, i.e. on the taxels. A plain convex hull bulged up to 24 mm over the pad, and the default decomposition still sat about 1.5 mm proud of the face. Regenerate the USD with `python tools/convert_urdf_to_usd.py`.
- **Observation:** `flexitac.FlexiTacSensor._normalize`, the real driver's normalization with threshold 25 and noise_scale 30. Contact frames are therefore peak-normalized to max = 1, exactly like real `lerobot_tactile` data.
- **Metadata:** the threshold, noise_scale and pad geometry are written to `info.json → tactile_sensors`.

## Differences to the real setup

- One fixed `top` camera at 640×480, no wrist camera. The real peg dataset uses `top` at 480×640, so the schemas match.
- `info.json` records `robot_type` as `isaacsim_so100_tactile_follower`.
- The tactile signal is geometric (distance-based); there is no elastomer mechanics.

## Dev tools

```bash
python tools/demo_tactile_static.py            # GUI: taxel positions on the pad (+ --check-host-penetration)
python tools/demo_tactile_vis.py --debug-vis   # analytic contact box sweeping the pad
python tools/demo_sinusoid.py --headless       # scene / joint sanity check without a leader
python tools/convert_urdf_to_usd.py            # regenerate the arm USD from the URDF (convex decomposition)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `SyntaxError` in `lerobot/motors/motors_bus.py` (or `pipeline.py`, …) | The Python 3.11 patch is not applied: `bash patches/apply_lerobot_py311.sh` |
| `No module named 'flexitac'` | `pip install -r requirements-isaacsim.txt` |
| `--robot.type=isaacsim_so100_tactile_follower` unknown | `pip install --no-deps -e .` in the same env (lerobot discovers `lerobot_robot_*` packages) |
| `Permission denied: '/dev/ttyACM0'` | Add yourself to `dialout` (see Installation) |
| Warning `send_action is called at ~N Hz but robot.fps=30` | Pass `--fps=30` to `lerobot-teleoperate` |
| First GUI start takes several minutes | Isaac Sim compiles RTX shaders once; later starts are fast |

## Assets

`assets/onshape_export/` holds the [onshape-to-robot](https://onshape-to-robot.readthedocs.io/) config for the modified gripper (see its README). API keys go into your environment or `~/.config/onshape-to-robot/config.json`, never into this repo (`.env` is git-ignored).
The arm USD in `src/lerobot_robot_isaacsim_tactile/assets/gripper_so100_tactile/usd/` was converted from the URDF with IsaacLab's URDF converter (settings in `usd/config.yaml`).

## License

Apache-2.0, the same license as [lerobot_tactile](https://github.com/TNA001-AI/lerobot_tactile); see [LICENSE](LICENSE). The vendored `warp_sdf_tactile` keeps its original IsaacLab BSD-3-Clause headers.
