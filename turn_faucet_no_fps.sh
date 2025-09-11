#!/bin/bash
sbatch --export=GROUP_NAME=no_FPS_TF,CONF_DIR=../slurm_confs/turn_faucet_no_fps,PPRL_ENV=turn_faucet run_6seeds.sbatch
