#!/bin/bash
sbatch --export=GROUP_NAME=FPS_TF_final,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/turn_faucet_no_fps,PPRL_ENV=turn_faucet /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run.sbatch
