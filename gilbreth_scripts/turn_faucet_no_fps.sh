#!/bin/bash
sbatch --export=GROUP_NAME=FPS_TF_final,CONF_DIR=../slurm_confs/turn_faucet_no_fps,PPRL_ENV=turn_faucet run_6seeds.sbatch
