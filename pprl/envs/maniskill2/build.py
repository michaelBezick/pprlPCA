from __future__ import annotations

import gymnasium as gym
import numpy as np

from pprl.envs.wrappers.continuous_task_wrapper import ContinuousTaskWrapper

from . import ManiSkillAddObsToInfoWrapper
from .add_part_id import AddPartIdWrapper
from .pointcloud_obs import PointCloudWrapper, SafePointCloudWrapper


def build(
    env_id: str,
    max_episode_steps: int | None,
    observation_type: str,
    env_kwargs: dict | None,
    pcd_kwargs: dict | None,
    min_num_points: int | None = None,
    continuous_task: bool = False,
    add_rendering_to_info: bool = False,
) -> gym.Env:
    observation_components = observation_type.split("+")


    if "pointcloud" in observation_components:
        maniskill_obs_mode = "pointcloud"
    elif "rgb" in observation_components or "rgbd" in observation_components:
        maniskill_obs_mode = "image"
    else:
        maniskill_obs_mode = "state"

    env_kwargs = env_kwargs or {}
    env = gym.make(
        env_id,
        obs_mode=maniskill_obs_mode,
        **env_kwargs,
    )

    if max_episode_steps is None:
        env = env.env  # unwrap once to remove Timeout wrapper
    else:
        env._max_episode_steps = max_episode_steps  # update default value

    # Environments are deterministic (always the same seed) unless explicitly seeded
    env.reset(seed=np.random.randint(1e9))

    if add_rendering_to_info:
        env = ManiSkillAddObsToInfoWrapper(env)

    if continuous_task:
        env = ContinuousTaskWrapper(env)

    if "pointcloud" in observation_components:
        pcd_kwargs = pcd_kwargs or {}

        if pcd_kwargs.get("exclude_handle_points", False) and (
            "camera_cfgs" not in env_kwargs
            or "add_segmentation" not in env_kwargs["camera_cfgs"]
        ):
            # if using segmentation-aware downsampling, add segmentation to rendering
            # NOTE: value of this key doesn't matter, the env only checks if the key exists
            env_kwargs["camera_cfgs"]["add_segmentation"] = True

        points_only = "state" not in observation_components
        points_only = True
        env = PointCloudWrapper(
            env,
            points_only=points_only,
            **pcd_kwargs,
        )

        if min_num_points is not None:
            env = SafePointCloudWrapper(env, min_num_points=min_num_points)

    if "id" in observation_components:
        env = AddPartIdWrapper(env)

    return env
