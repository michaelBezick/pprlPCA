#!/bin/bash
sbatch --export=GROUP_NAME=DR_TF_final,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/turn_faucet_dr,PPRL_ENV=turn_faucet /home/mbezick/Repos/pprlPCA/gilbreth_scripts/run_6seeds.sbatch
