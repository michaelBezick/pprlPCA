#!/bin/bash
sbatch --export=GROUP_NAME=OCDoor_baseline,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/open_cabinet_door_baseline,PPRL_ENV=open_cabinet_door /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run.sbatch
