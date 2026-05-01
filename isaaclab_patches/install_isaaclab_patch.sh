#!/usr/bin/env bash
# install_isaaclab_patch.sh
# Apply the warp_sdf_tactile sensor patch into an existing IsaacLab installation.
# Usage:
#   ./install_isaaclab_patch.sh /path/to/IsaacLab
#
# This script:
#   1. Copies warp_sdf_tactile/ sensor files into IsaacLab's sensor directory
#   2. Appends the import line to isaaclab/sensors/__init__.py if not already present

set -e

ISAACLAB_DIR="${1:-}"

if [[ -z "$ISAACLAB_DIR" ]]; then
    echo "Usage: $0 /path/to/IsaacLab"
    exit 1
fi

SENSORS_DIR="$ISAACLAB_DIR/source/isaaclab/isaaclab/sensors"

if [[ ! -d "$SENSORS_DIR" ]]; then
    echo "ERROR: Cannot find isaaclab sensors dir at: $SENSORS_DIR"
    echo "Make sure the path points to the root of your IsaacLab clone."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/sensors/warp_sdf_tactile"

echo "[1/2] Copying warp_sdf_tactile sensor files..."
cp -r "$PATCH_DIR" "$SENSORS_DIR/warp_sdf_tactile"
echo "      -> $SENSORS_DIR/warp_sdf_tactile/"

INIT_FILE="$SENSORS_DIR/__init__.py"
IMPORT_LINE="from .warp_sdf_tactile import *  # noqa: F401, F403"

echo "[2/2] Patching $INIT_FILE ..."
if grep -qF "warp_sdf_tactile" "$INIT_FILE"; then
    echo "      Already patched, skipping."
else
    echo "" >> "$INIT_FILE"
    echo "# --- sim_tactile patch ---" >> "$INIT_FILE"
    echo "$IMPORT_LINE" >> "$INIT_FILE"
    echo "      Done."
fi

echo ""
echo "Patch applied successfully!"
echo "You can verify by running:"
echo "  python -c 'from isaaclab.sensors import WarpSdfTactileSensor; print(WarpSdfTactileSensor)'"
