#!/bin/bash
sbatch --export=GROUP_NAME=PCA_TF,CONF_DIR=../slurm_confs/turn_faucet_pca,PPRL_ENV=turn_faucet run_6seeds.sbatch
