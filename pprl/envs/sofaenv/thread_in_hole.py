from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
from sofa_env.scenes.thread_in_hole.thread_in_hole_env import (
    ActionType,
    ObservationType,
    RenderMode,
    ThreadInHoleEnv,
)

from . import add_env_wrappers, convert_to_array


def build(
    max_episode_steps: int,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    observation_type: Literal["pointcloud", "rgb", "rgbd"],
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    insertion_ratio_threshold: float,
    settle_steps: int,
    simple_success_check: bool,
    camera_reset_noise: list | None,
    camera_rotation_reset_noise_deg: list | None,
    hole_rotation_reset_noise: list | None,
    hole_position_reset_noise: list | None,
    reward_amount_dict: dict,
    create_scene_kwargs: dict | None = None,
    add_rendering_to_info: bool = False,
    **kwargs,
) -> gym.Env:
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore

    if camera_reset_noise is not None:
        camera_reset_noise = np.asarray(camera_reset_noise)
    if camera_rotation_reset_noise_deg is not None:
        camera_rotation_reset_noise_deg = np.asarray(camera_rotation_reset_noise_deg)
    if hole_rotation_reset_noise is not None:
        hole_rotation_reset_noise = np.asarray(hole_rotation_reset_noise)
    if hole_position_reset_noise is not None:
        hole_position_reset_noise = np.asarray(hole_position_reset_noise)

    if observation_type in ("pointcloud", "rgb"):
        obs_type = ObservationType.RGB
    elif observation_type == "rgbd":
        obs_type = ObservationType.RGBD
    else:
        raise ValueError(f"Invalid observation type: {observation_type}")

    if create_scene_kwargs is not None:
        convert_to_array(create_scene_kwargs)

    env = ThreadInHoleEnv(
        observation_type=obs_type,
        render_mode=render_mode,
        action_type=action_type,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        settle_steps=settle_steps,
        create_scene_kwargs=create_scene_kwargs,
        reward_amount_dict=reward_amount_dict,
        camera_reset_noise=camera_reset_noise,
        camera_rotation_reset_noise_deg=camera_rotation_reset_noise_deg,
        hole_rotation_reset_noise=hole_rotation_reset_noise,
        hole_position_reset_noise=hole_position_reset_noise,
        insertion_ratio_threshold=insertion_ratio_threshold,
        simple_success_check=simple_success_check,
    )

    env = add_env_wrappers(
        env,
        max_episode_steps=max_episode_steps,
        add_rendering_to_info=add_rendering_to_info,
        observation_type=observation_type,
        **kwargs,
    )

    return env
