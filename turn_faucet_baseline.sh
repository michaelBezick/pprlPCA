#!/bin/bash
sbatch --export=GROUP_NAME=baseline_TF,CONF_DIR=../slurm_confs/turn_faucet_baseline,PPRL_ENV=turn_faucet run_6seeds.sbatch
