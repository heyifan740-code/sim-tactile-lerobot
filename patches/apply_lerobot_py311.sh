#!/usr/bin/env bash
# Apply the Python 3.11 compatibility patch to the lerobot_tactile submodule (idempotent).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE="$ROOT/third_party/lerobot_tactile"
PATCH="$ROOT/patches/lerobot_tactile_py311.patch"

if git -C "$SUBMODULE" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "py311 patch already applied."
else
    git -C "$SUBMODULE" apply "$PATCH"
    echo "py311 patch applied to $SUBMODULE"
fi
