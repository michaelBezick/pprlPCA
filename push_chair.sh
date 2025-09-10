#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if (( $# != 2 )); then
  echo "Error, usage: bash thread_in_hole.sh [num_runs] \"group_name\" "
  exit
fi

num_runs=$1
group_name=$2

if (( $num_runs > 8 )); then
  echo "Too many runs"
  exit
fi

ulimit -n 4096

log_dir="./logs/$group_name"
mkdir -p "$log_dir"

gpu_id=0
run_number=0

while (( gpu_id < 4)); do


  log_file="$log_dir/run_${run_number}.log"

  echo "$gpu_id"

  CUDA_VISIBLE_DEVICES=$gpu_id nohup python scripts/maniskill_train_sac.py wandb.group_name="$group_name" env=push_chair model=ppt > "$log_file" 2>&1 &

  ((run_number++))
  gpu_id=$(( run_number % 4 ))

  if (( run_number == $num_runs )); then
    exit
  fi

done
