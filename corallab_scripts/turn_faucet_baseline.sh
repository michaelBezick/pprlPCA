#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [ "$#" -ne 1 ]; then
  echo "Error: Need GPU ID."
  exit 1
fi

echo "GPU ID: $1"

nohup env GROUP_NAME=PC_PCA CUDA_VISIBLE_DEVICES="$1" python ./scripts/train_sac.py \
  -cp "../slurm_confs/turn_faucet_baseline" env=turn_faucet model=pointgpt_rl algo=aux_sac \
  > train.log 2>&1 &
disown
