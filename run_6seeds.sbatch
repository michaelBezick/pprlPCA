#!/bin/bash
#SBATCH -J ms2_ppt_seeds
#SBATCH -A kildisha
#SBATCH -p a100-80gb
#SBATCH --qos=normal
#SBATCH --array=0-5%1

# -- key fixes --
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1          # (or use: #SBATCH --gres=gpu:1)
#SBATCH --cpus-per-task=10
# ----------------

#SBATCH --mem=20G
#SBATCH --time=48:00:00

#SBATCH -o logs/%x/%A_%a.out
#SBATCH -e logs/%x/%A_%a.err

# ---------------- user env ----------------
set -euo pipefail
ulimit -n 4096

cd /home/mbezick/Repos/pprlPCA
module load conda
conda activate pprl_quat

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

# ---------------- parameters ----------------
SEED="${SLURM_ARRAY_TASK_ID}"
GROUP_NAME="${GROUP_NAME:-my_experiment}"

mkdir -p "logs/${GROUP_NAME}"

echo "Job ${SLURM_JOB_ID} / array task ${SEED} on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---------------- run ----------------
srun --unbuffered python scripts/maniskill_train_sac2.py \
  -cp "$CONF_DIR" \
  wandb.group_name="${GROUP_NAME}" \
  seed="${SEED}" \
  env="${PPRL_ENV}" \
  model=ppt
