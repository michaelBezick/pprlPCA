#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

if (( $# < 3 )); then
  echo "Usage: $0 <num_seeds> \"group_name\" <gpu_id1> [gpu_id2 ... gpu_idN]"
  exit 1
fi

num_seeds="$1"; shift
group_name="$1"; shift
gpu_ids=( "$@" )

if (( ${#gpu_ids[@]} != num_seeds )); then
  echo "Error: you passed ${#gpu_ids[@]} GPU id(s) but num_seeds=${num_seeds}"
  exit 1
fi

ulimit -n 4096

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="./logs/${group_name}_${timestamp}"
mkdir -p "$log_dir"

echo "Launching ${num_seeds} runs under W&B group \"${group_name}\""
echo "Logs: ${log_dir}"

for (( i=0; i<num_seeds; i++ )); do
  gpu="${gpu_ids[$i]}"
  log_file="${log_dir}/run_${i}.log"

  echo "[RUN ${i}] -> GPU=${gpu}  |  log=${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
    nohup python scripts/multi_eval_train_sac.py \
      parallel=False \
      wandb.group_name="${group_name}" \
      env=thread_in_hole \
      model=pointgpt_rl \
      algo=aux_sac \
      env.image_shape="[64, 64]" \
      runner.n_steps=450000 \
      > "${log_file}" 2>&1 &
done

echo "All ${num_seeds} runs launched. Waiting for them to finish..."
wait
echo "Done."

