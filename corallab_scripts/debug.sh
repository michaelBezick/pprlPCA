#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
ulimit -n 4096
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python ./scripts/train_sac.py -cp "../slurm_confs/turn_faucet_pca" parallel=False env=turn_faucet model=ppt platform=debug batch_B=1 eval.n_eval_envs=1 eval.max_trajectories=1 runner.log_interval_steps=1
