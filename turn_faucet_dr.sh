#!/bin/bash
sbatch --export=GROUP_NAME=DR_TF_final,CONF_DIR=../slurm_confs/turn_faucet_dr,PPRL_ENV=turn_faucet run_6seeds.sbatch
