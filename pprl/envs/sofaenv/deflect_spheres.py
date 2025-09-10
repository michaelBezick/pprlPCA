from __future__ import annotations

from typing import Literal, Sequence

import gymnasium as gym
import numpy as np
from sofa_env.scenes.deflect_spheres.deflect_spheres_env import (
    ActionType,
    DeflectSpheresEnv,
    ObservationType,
    RenderMode,
)

from . import add_env_wrappers, convert_to_array


def build(
    max_episode_steps: int,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    observation_type: Literal["pointcloud", "rgb", "rgbd"],
    image_shape: Sequence[int],
    frame_skip: int,
    time_step: float,
    settle_steps: int,
    reward_amount_dict: dict,
    single_agent: bool,
    num_objects: int,
    num_deflect_to_win: int,
    min_deflection_distance: float,
    create_scene_kwargs: dict | None = None,
    camera_reset_noise: list | None = None,
    camera_rotation_reset_noise_deg: list | None = None,
    add_rendering_to_info: bool = False,
    **kwargs,
) -> gym.Env:
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore

    if camera_reset_noise is not None:
        camera_reset_noise = np.asarray(camera_reset_noise)

    if observation_type in ("pointcloud", "rgb"):
        obs_type = ObservationType.RGB
    elif observation_type == "rgbd":
        obs_type = ObservationType.RGBD
    else:
        raise ValueError(f"Invalid observation type: {observation_type}")

    if create_scene_kwargs is not None:
        convert_to_array(create_scene_kwargs)

    env = DeflectSpheresEnv(
        observation_type=obs_type,
        render_mode=render_mode,
        action_type=action_type,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        settle_steps=settle_steps,
        single_agent=single_agent,
        num_objects=num_objects,
        num_deflect_to_win=num_deflect_to_win,
        min_deflection_distance=min_deflection_distance,
        create_scene_kwargs=create_scene_kwargs,
        reward_amount_dict=reward_amount_dict,
        camera_reset_noise=camera_reset_noise,
        camera_rotation_reset_noise_deg=camera_rotation_reset_noise_deg,
    )
    env = add_env_wrappers(
        env,
        max_episode_steps=max_episode_steps,
        add_rendering_to_info=add_rendering_to_info,
        observation_type=observation_type,
        **kwargs,
    )

    return env
