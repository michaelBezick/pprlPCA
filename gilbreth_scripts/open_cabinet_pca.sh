#!/bin/bash
sbatch --export=GROUP_NAME=PCA_OCD_no_extra,CONF_DIR=../slurm_confs/open_cabinet_drawer_pca,PPRL_ENV=open_cabinet_drawer /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run_6seeds.sbatch
