"""Evaluate a trained lerobot policy inside IsaacSim using our sim follower.

Mirrors the real-robot eval flow: load policy checkpoint, roll out for N
episodes against the `IsaacSimSO100TactileFollower`, report success rate
+ average completion time + contact/tactile stats.
"""

from __future__ import annotations

import main_package.sim_tactile.follower  # noqa: F401  (draccus registration)


def _main() -> None:
    # TODO(impl): wire up lerobot eval.
    #
    #   from lerobot.policies.factory import make_policy
    #   from main_package.sim_tactile.follower import (
    #       IsaacSimSO100TactileFollower, IsaacSimSO100TactileFollowerConfig,
    #   )
    #
    #   # (parse CLI via draccus)
    #   robot = IsaacSimSO100TactileFollower(cfg.robot)
    #   policy = make_policy(cfg.policy, ...)
    #   with robot:
    #       for ep in range(cfg.num_episodes):
    #           obs = robot.get_observation()
    #           done = False
    #           while not done:
    #               action = policy.select_action(obs)
    #               robot.send_action(action)
    #               obs = robot.get_observation()
    #               done = termination_fn(obs)  # e.g. peg below hole rim
    #           ...
    raise NotImplementedError("eval_policy_sim entrypoint is a skeleton")


if __name__ == "__main__":
    _main()
