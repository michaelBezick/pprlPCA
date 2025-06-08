#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if (( $# != 2 )); then
  echo "Error, usage: bash thread_in_hole.sh [num_runs] \"group_name\" "
  exit
fi

num_runs=$1
log_filename=$2

if (( $num_runs > 6 )); then
  echo "Too many runs"
  exit
fi

ulimit -n 4096

gpu_id=0
run_number=0

while (( gpu_id < 3)); do
  CUDA_VISIBLE_DEVICES=$gpu_id nohup python scripts/train_sac.py wandb.group_name="$log_filename" env=thread_in_hole model=pointgpt_rl algo=aux_sac  \
  env.image_shape="[64, 64]" runner.n_steps=2000000 > "./logs/$log_filename.log" 2>&1 &


  ((run_number++))

  if (( run_number == $num_runs )); then
    exit
  fi

  if (( run_number % 2 == 0 )); then
    ((gpu_id++))
  fi

done
