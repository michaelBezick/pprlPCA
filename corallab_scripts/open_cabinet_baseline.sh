#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

nohup env GROUP_NAME=OCD_Baseline CUDA_VISIBLE_DEVICES=0 python ./scripts/train_sac.py \
  -cp "../slurm_confs/open_cabinet_drawer_baseline" env=open_cabinet_drawer model=pointgpt_rl algo=aux_sac \
  > train.log 2>&1 &
disown
