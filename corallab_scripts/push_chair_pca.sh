#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

nohup env GROUP_NAME=PC_PCA CUDA_VISIBLE_DEVICES=0 python ./scripts/train_sac.py \
  -cp "../slurm_confs/push_chair_pca" env=push_chair model=pointgpt_rl algo=aux_sac \
  > train.log 2>&1 &
disown
