#!/bin/bash
sbatch --export=GROUP_NAME=PC_baseline,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/push_chair_baseline,PPRL_ENV=push_chair /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run.sbatch
