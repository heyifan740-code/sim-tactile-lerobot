# sim-tactile-lerobot

An Isaac Sim + LeRobot teleoperation data collection system with tactile sensing.

Enables a physical **SO100 leader arm** to teleoperate a simulated **SO100 follower** with
tactile pad sensors in Isaac Sim, recording demonstrations in the lerobot dataset format.

## Architecture

```
IsaacLab (Isaac Sim physics)
└── WarpSdfTactileSensor  ← custom sensor (installed via patch)
    └── main_package/sim_tactile/
        ├── scene/              ← Isaac Sim scene setup
        ├── sensors/            ← tactile pad configuration
        ├── follower/           ← lerobot Robot wrapper (IsaacSimSO100TactileFollower)
        ├── scripts/
        │   └── teleop_record.py  ← main entry point
        ├── assets/             ← USD / URDF robot models
        └── _vendor/
            └── lerobot_tactile/  ← git submodule (lerobot fork with tactile support)
```

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Isaac Sim | 4.5 | via Omniverse Launcher or Docker |
| IsaacLab | ≥ 2.0 | see [installation](https://isaac-sim.github.io/IsaacLab/) |
| Python | 3.10–3.11 | bundled with IsaacLab |
| git-lfs | any | for USD/URDF assets |

---

## Installation

### Step 1 — Clone this repo

```bash
git clone --recurse-submodules https://github.com/YOUR_ORG/sim-tactile-lerobot.git
cd sim-tactile-lerobot
```

> **Important**: `--recurse-submodules` fetches `lerobot_tactile` automatically.  
> If you forgot it, run: `git submodule update --init --recursive`

If assets are tracked with Git LFS:

```bash
git lfs install
git lfs pull
```

### Step 2 — Clone and install IsaacLab

Follow the [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/workspace/IsaacLab
cd ~/workspace/IsaacLab
./isaaclab.sh --install
```

### Step 3 — Apply the WarpSdfTactile sensor patch to IsaacLab

This repo includes a custom GPU-accelerated tactile sensor (`WarpSdfTactileSensor`) that
needs to be installed into your IsaacLab checkout.

```bash
# From the sim-tactile-lerobot root:
bash isaaclab_patches/install_isaaclab_patch.sh ~/workspace/IsaacLab
```

Verify the patch:

```bash
~/workspace/IsaacLab/isaaclab.sh -p - <<'EOF'
from isaaclab.sensors import WarpSdfTactileSensor
print("WarpSdfTactileSensor OK:", WarpSdfTactileSensor)
EOF
```

### Step 4 — Install lerobot_tactile

The `lerobot_tactile` submodule is a fork of LeRobot with tactile observation support.

```bash
cd main_package/sim_tactile/_vendor/lerobot_tactile
pip install -e ".[feetech]"
```

> The `lerobot_tactile` package installs as `lerobot` (it replaces the base package).
> Do this in the same Python environment used by IsaacLab (i.e., after running
> `~/workspace/IsaacLab/isaaclab.sh` to activate the environment).

### Step 5 — Set workspace path

The entry script auto-discovers `lerobot_dev` via the environment variable or relative paths.
Point it to the `lerobot_tactile` src:

```bash
export LEROBOT_DEV_SRC=$(pwd)/main_package/sim_tactile/_vendor/lerobot_tactile/src
```

Or add to your `~/.bashrc`:

```bash
export LEROBOT_DEV_SRC=/path/to/sim-tactile-lerobot/main_package/sim_tactile/_vendor/lerobot_tactile/src
```

---

## Running Teleoperation + Data Collection

### Hardware setup

Connect the physical SO100 **leader arm** to USB:

```bash
ls /dev/ttyACM*   # find the port
```

### Launch

From the **IsaacLab root** (`~/workspace/IsaacLab`):

```bash
./isaaclab.sh -p /path/to/sim-tactile-lerobot/main_package/sim_tactile/scripts/teleop_record.py \
    --robot.type=isaacsim_so100_tactile_follower \
    --robot.urdf_path=/path/to/sim-tactile-lerobot/main_package/sim_tactile/assets/gripper_so100_tactile/urdf/so100_follower_tactile.urdf \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --display_data=true \
    --dataset.repo_id=local/sim_so100_tactile_demo \
    --dataset.num_episodes=5 \
    --dataset.single_task="peg insertion"
```

> **Tip**: Replace `/path/to/sim-tactile-lerobot` with the absolute path to your clone.
> If `sim-tactile-lerobot` is placed as a sibling of `IsaacLab` (i.e., both in `~/workspace/`),
> the relative path `../sim-tactile-lerobot/...` also works.

### Key CLI options

| Option | Description |
|---|---|
| `--robot.urdf_path` | Path to the SO100 tactile URDF |
| `--teleop.port` | USB serial port of the leader arm |
| `--dataset.repo_id` | Output dataset identifier (`local/<name>`) |
| `--dataset.num_episodes` | Number of demos to record |
| `--dataset.single_task` | Task description string |
| `--display_data` | Show live tactile visualization |

---

## Dataset output

Recorded datasets are stored in `~/.cache/huggingface/lerobot/<dataset.repo_id>/`
in the standard LeRobot dataset format (HDF5 + metadata).

To upload to HuggingFace Hub:

```bash
huggingface-cli upload <your_hf_org>/sim_so100_tactile_demo \
    ~/.cache/huggingface/lerobot/local/sim_so100_tactile_demo
```

---

## Tactile sensor notes

The `WarpSdfTactileSensor` uses Warp GPU kernels for real-time SDF queries against
the gripper mesh. Key configuration parameters (in `sensors/pad_on_wrist_roll.py`):

| Parameter | Value | Meaning |
|---|---|---|
| `DEFAULT_TACTILE_NORMAL_OFFSET_M` | `-0.001` m | Taxel sampling layer (negative = 1mm inside pad surface) |
| `mesh_contact_onset_m` | `0.001` m | Soft-contact onset: signal starts when object is within 1mm of surface |
| Grid size | 12 × 32 | Matches physical SO100 tactile sensor |

---

## Troubleshooting

**`WarpSdfTactileSensor` not found**  
→ Re-run `bash isaaclab_patches/install_isaaclab_patch.sh ~/workspace/IsaacLab`

**`--robot.type=isaacsim_so100_tactile_follower` not recognized**  
→ Ensure `lerobot_tactile` is installed and `LEROBOT_DEV_SRC` is set correctly.

**Tactile signal is zero / intermittent**  
→ Check that `mesh_contact_onset_m=0.001` is set in `sensors/pad_on_wrist_roll.py`.

**Isaac Sim crashes on startup**  
→ Verify Isaac Sim 4.5 is installed and the Nucleus server is running.

---

## License

See [LICENSE](LICENSE).
