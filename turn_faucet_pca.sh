#!/bin/bash
sbatch --export=GROUP_NAME=PCA_TF_final,CONF_DIR=../slurm_confs/turn_faucet_pca,PPRL_ENV=turn_faucet run_6seeds.sbatch
