#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

nohup env CUDA_VISIBLE_DEVICES=0 python ./scripts/train_sac.py \
  -cp "../slurm_confs/push_chair_baseline" env=push_chair model=pointgpt_rl algo=aux_sac \
  > train.log 2>&1 &
disown
