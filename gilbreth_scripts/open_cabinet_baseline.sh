#!/bin/bash
sbatch --export=GROUP_NAME=baseline_OCD_no_extra,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/open_cabinet_drawer_baseline,PPRL_ENV=open_cabinet_drawer /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run_6seeds.sbatch
