#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [ "$#" -ne 1 ]; then
  echo "Error: Need GPU ID."
  exit 1
fi

echo "GPU ID: $1"

nohup env GROUP_NAME=OCD_Baseline CUDA_VISIBLE_DEVICES="$1" python ./scripts/train_sac.py \
  -cp "../slurm_confs/open_cabinet_drawer_baseline" env=open_cabinet_drawer model=ppt  \
  > train.log 2>&1 &
disown
