#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
ulimit -n 4096
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 GROUP_NAME=DEBUG python ./scripts/train_sac.py -cp "../slurm_confs/open_cabinet_door_baseline" parallel=False env=open_cabinet_door model=ppt platform=debug batch_B=1 eval.n_eval_envs=1 eval.max_trajectories=1 runner.log_interval_steps=1 wandb.group_name=DEBUG
