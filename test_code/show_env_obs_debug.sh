#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
ulimit -n 4096
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python scripts/my_show_env_obs.py env=thread_in_hole \
env.create_scene_kwargs.camera_config.placement_kwargs.position="[100, 0, 200.0]" \
env.create_scene_kwargs.camera_config.placement_kwargs.lookAt="[10., 0., 55.]"
