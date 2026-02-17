#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

nohup env GROUP_NAME=PC_Baseline CUDA_VISIBLE_DEVICES=2 python ./scripts/train_sac.py \
  -cp "../slurm_confs/push_chair_baseline" env=push_chair model=pointgpt_rl algo=aux_sac \
  > train.log 2>&1 &
disown
