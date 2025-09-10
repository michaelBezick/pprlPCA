#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export WANDB_DISABLED=true
ulimit -n 4096
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python scripts/train_sac.py env.render_mode=remote env=thread_in_hole model=pointgpt_rl algo=aux_sac platform=debug batch_B=1 eval.n_eval_envs=1 eval.max_trajectories=1 parallel=False
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python scripts/truly_og_train_sac.py parallel=False env=push_chair model=ppt platform=debug batch_B=1 runner.n_steps=1 eval.n_eval_envs=1 eval.max_trajectories=1
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python scripts/original_train_sac.py parallel=False env=deflect_spheres model=ppt platform=debug batch_B=1 runner.n_steps=1 eval.n_eval_envs=1 eval.max_trajectories=1
