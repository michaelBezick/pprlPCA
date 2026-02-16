#!/bin/bash
sbatch --export=GROUP_NAME=OCD_baseline,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/open_cabinet_drawer_baseline,PPRL_ENV=open_cabinet_drawer /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run.sbatch
