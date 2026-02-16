#!/bin/bash
sbatch --export=GROUP_NAME=OCD_PCA,CONF_DIR=../slurm_confs/open_cabinet_drawer_pca,PPRL_ENV=open_cabinet_drawer /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run_6seeds.sbatch
