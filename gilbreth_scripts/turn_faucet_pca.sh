#!/bin/bash
sbatch --export=GROUP_NAME=PCA_TF_no_extra,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/turn_faucet_pca,PPRL_ENV=turn_faucet run_6seeds.sbatch
