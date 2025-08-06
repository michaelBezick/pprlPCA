#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if (( $# != 2 )); then
  echo "Error, usage: bash thread_in_hole.sh [num_runs] \"group_name\" "
  exit
fi

num_runs=$1
group_name=$2

if (( $num_runs > 6 )); then
  echo "Too many runs"
  exit
fi

ulimit -n 4096

log_dir="./logs/$group_name"
mkdir -p "$log_dir"

gpu_id=0
run_number=0

while (( gpu_id < 3)); do


  log_file="$log_dir/run_${run_number}.log"


  CUDA_VISIBLE_DEVICES=$gpu_id nohup python scripts/multi_eval_train_sac.py parallel=True wandb.group_name="$group_name" env=thread_in_hole model=pointgpt_rl algo=aux_sac  \
  env.image_shape="[64, 64]" runner.n_steps=450000 > "$log_file" 2>&1 &


  ((run_number++))

  if (( run_number == $num_runs )); then
    exit
  fi

  if (( run_number % 2 == 0 )); then
    ((gpu_id++))
  fi

done
