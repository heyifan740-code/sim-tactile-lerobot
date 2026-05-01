# Onshape export — SO100 follower with tactile pad

Source document: <https://cad.onshape.com/documents/04c29f787d3b54fbfb9a6525/w/c9d62c1d4c7b79404435ea6b/e/8f83782a5cf94ceea58f536c>

## One-time setup

1. Get an API key/secret at <https://dev-portal.onshape.com/keys> (free).
2. Put them in your shell env (preferred) or a config file:

```bash
# either (shell env)
export ONSHAPE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
export ONSHAPE_API_SECRET=yyyyyyyyyyyyyyyyyyyyyyyy

# or (persistent file — chmod 600)
mkdir -p ~/.config/onshape-to-robot
cat > ~/.config/onshape-to-robot/config.json <<'EOF'
{ "accessKey": "xxxxxxxxxxxx", "secretKey": "yyyyyyyyyyyy" }
EOF
chmod 600 ~/.config/onshape-to-robot/config.json
```

**Do not paste the keys into this repo or into the chat.**

## Run export

```bash
cd /home/fan/workspace/IsaacLab/main_package/sim_tactile/assets/onshape_export
onshape-to-robot .
```

This reads `config.json` in the cwd, hits the Onshape API, and writes:

- `robot.urdf`
- `*.stl` mesh files (one per Onshape part)
- `*.scad` (ignore)

## After export

Move the useful pieces into the project's asset tree:

```bash
mkdir -p ../gripper_so100_tactile/{urdf,meshes}
mv robot.urdf ../gripper_so100_tactile/urdf/so100_follower_tactile.urdf
mv *.stl      ../gripper_so100_tactile/meshes/
```

Then inspect the URDF and verify:

- `Wrist_Roll_08c` is present as a link (this is where the tactile pad mounts).
- The fixed-base link is the shoulder mount / table interface.

If `Wrist_Roll_08c` got merged into a parent by the simplifier, set in
`config.json`:

```json
"noFrames": ["Wrist_Roll_08c"]
```

…and re-export so that link is preserved.
