#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export WANDB_DISABLED=true
ulimit -n 4096
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python scripts/train_sac.py env.render_mode=remote env=thread_in_hole model=pointgpt_rl algo=aux_sac platform=debug batch_B=1 eval.n_eval_envs=1 eval.max_trajectories=1 parallel=False
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python ./scripts/train_sac.py -cp "../slurm_confs/turn_faucet_pca" parallel=False env=turn_faucet model=ppt platform=debug batch_B=1 runner.n_steps=1 eval.n_eval_envs=1 eval.max_trajectories=1 runner.log_interval_steps=1
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python ./scripts/train_sac.py -cp "../slurm_confs/turn_faucet_baseline" parallel=False env=turn_faucet model=ppt platform=debug batch_B=1 runner.n_steps=1 eval.n_eval_envs=1 eval.max_trajectories=1
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python ./scripts/train_sac.py parallel=False env=open_cabinet_drawer model=ppt platform=debug batch_B=1 runner.n_steps=1 eval.n_eval_envs=1 eval.max_trajectories=1
