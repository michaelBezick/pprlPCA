#!/bin/bash
sbatch --export=GROUP_NAME=TF_PCA,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/turn_faucet_pca,PPRL_ENV=turn_faucet /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run.sbatch
