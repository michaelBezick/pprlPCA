#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export WANDB_DISABLED=true
ulimit -n 4096
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python scripts/train_sac.py env=thread_in_hole model=pointgpt_rl algo=aux_sac platform=debug batch_B=1 eval.n_eval_envs=1 eval.max_trajectories=1 parallel=False
