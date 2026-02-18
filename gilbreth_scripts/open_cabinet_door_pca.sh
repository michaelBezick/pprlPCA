#!/bin/bash
sbatch --export=GROUP_NAME=OCDoor_PCA,CONF_DIR=../slurm_confs/open_cabinet_door_pca,PPRL_ENV=open_cabinet_door /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run.sbatch
