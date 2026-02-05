#!/bin/bash
sbatch --export=GROUP_NAME=no_FPS_OCD_final,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/open_cabinet_drawer_no_fps,PPRL_ENV=open_cabinet_drawer /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run_6seeds.sbatch
