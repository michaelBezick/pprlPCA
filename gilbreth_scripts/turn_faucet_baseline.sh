#!/bin/bash
sbatch --export=GROUP_NAME=baseline_TF_no_extra,CONF_DIR=/home/mbezick/Repos/pprlPCA/slurm_confs/turn_faucet_baseline,PPRL_ENV=turn_faucet run_6seeds.sbatch
