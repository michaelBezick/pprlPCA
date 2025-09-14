#!/bin/bash
#SBATCH -J plswork
#SBATCH -A kildisha
#SBATCH -p training
#SBATCH --qos=training

# -- key fixes --
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
# ----------------

#SBATCH --mem=100G
#SBATCH --time=24:00:00

#SBATCH -o logs/%x/%A_%a.out
#SBATCH -e logs/%x/%A_%a.err

# ---------------- user env ----------------
set -e -o pipefail
ulimit -n 4096

cd /home/mbezick/Repos/pprlPCA
module load conda/2025.09
conda activate pprl_quat

cd /scratch/gilbreth/mbezick/pprlPCA2

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

# ---------------- parameters ----------------
SEED="${SLURM_ARRAY_TASK_ID}"
GROUP_NAME="${GROUP_NAME:-my_experiment}"

mkdir -p "logs/${GROUP_NAME}"

echo "Job ${SLURM_JOB_ID} / array task ${SEED} on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---------------- run ----------------
srun --unbuffered --export=ALL python scripts/maniskill_train_sac2.py \
  -cp "$CONF_DIR" \
  wandb.group_name="${GROUP_NAME}" \
  env="${PPRL_ENV}" \
  model=pointgpt_rl algo=aux_sac
