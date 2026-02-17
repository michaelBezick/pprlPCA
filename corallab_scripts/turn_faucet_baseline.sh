#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

nohup env GROUP_NAME=PC_PCA CUDA_VISIBLE_DEVICES=1 python ./scripts/train_sac.py \
  -cp "../slurm_confs/turn_faucet_baseline" env=turn_faucet model=pointgpt_rl algo=aux_sac \
  > train.log 2>&1 &
disown
