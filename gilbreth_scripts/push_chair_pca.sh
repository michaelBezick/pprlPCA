#!/bin/bash
sbatch --export=GROUP_NAME=PC_PCA,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/push_chair_pca,PPRL_ENV=push_chair /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run.sbatch
