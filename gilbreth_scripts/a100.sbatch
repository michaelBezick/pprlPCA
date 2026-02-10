#!/bin/bash
#SBATCH -J ms2_ppt_seeds_a100
#SBATCH -A kildisha
#SBATCH -p a100-80gb
#SBATCH --qos=normal

# one node, ONE GPU total; run 3 tasks that all share that GPU
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16      # total CPUs reserved = 3 * 16 = 48

# max CPU memory available on a100-80gb nodes (leave as you had it)
#SBATCH --mem=300G
#SBATCH --time=48:00:00

# wrapper logs for the whole job
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err

# --------------- env ---------------
set -e -o pipefail
ulimit -n 4096

module load conda/2025.09
conda activate pprl_quat
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# paths
cd /home/mbezick/Repos/pprlPCA
cd /scratch/gilbreth/mbezick/pprlPCA2

# two group names (override with env vars if you want)
GROUP_NAME_A="baseline_OCD_more_mismatch"
GROUP_NAME_B="PCA_OCD_more_mismatch"

# create log dirs for both groups
mkdir -p "logs/${GROUP_NAME_A}" "logs/${GROUP_NAME_B}"

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (all 3 tasks share this GPU)"
echo "Groups: A='${GROUP_NAME_A}' (2 runs), B='${GROUP_NAME_B}' (1 run)"

# --------------- run (3 tasks share 1 GPU) ---------------
# Per-task logs (separate files) in addition to the wrapper logs above
srun -n "${SLURM_NTASKS}" \
  --output="logs/%x/%j_task%t.out" \
  --error="logs/%x/%j_task%t.err" \
  --export=ALL \
  bash -lc '
    # Map tasks: 0,1 -> GROUP_NAME_A; 2 -> GROUP_NAME_B
    if [ "$SLURM_PROCID" -lt 2 ]; then
        GROUP="${GROUP_NAME_A}"
      
        mkdir -p "logs/${GROUP}"
        echo "[Task ${SLURM_PROCID}] group=${GROUP}"

        python scripts/train_sac.py \
          -cp "../slurm_confs/open_cabinet_drawer_baseline" \
          wandb.group_name="baseline_OCD_pls" \
          env="open_cabinet_drawer" \
          model=pointgpt_rl \
          algo=aux_sac
    else
        GROUP="${GROUP_NAME_B}"
        mkdir -p "logs/${GROUP}"
        echo "[Task ${SLURM_PROCID}] group=${GROUP}"

        python scripts/train_sac.py \
          -cp "../slurm_confs/open_cabinet_drawer_pca" \
          wandb.group_name="PCA_OCD_pls" \
          env="open_cabinet_drawer" \
          model=pointgpt_rl \
          algo=aux_sac
    fi
  '

