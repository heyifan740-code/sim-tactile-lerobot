"""sim_tactile — IsaacSim backend for the lerobot_tactile (new-sync) workflow.

Design: expose an IsaacSim-backed lerobot `Robot` subclass so that the existing
real-robot teleop / record / train / eval scripts from lerobot_tactile work
unchanged, with the sim follower swapped in via draccus config.

Layout
------
- follower/   IsaacSim follower as a lerobot Robot
- scene/      Table + fixed-base arm + socket scene builder
- sensors/    Warp SDF tactile pad attached to Wrist_Roll_08c
- scripts/    Teleop-record and sim-eval entrypoints
- assets/     URDF/USD/STL assets (gripper, peg, hole)
"""
