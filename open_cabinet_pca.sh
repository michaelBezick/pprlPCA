#!/bin/bash
sbatch --export=GROUP_NAME=PCA_OCD,CONF_DIR=../slurm_confs/open_cabinet_drawer_pca,PPRL_ENV=open_cabinet_drawer run_6seeds.sbatch
