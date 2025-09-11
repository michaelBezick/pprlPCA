#!/bin/bash
sbatch --export=GROUP_NAME=no_FPS_OCD,CONF_DIR=../slurm_confs/open_cabinet_drawer_no_fps,PPRL_ENV=open_cabinet_drawer run_6seeds.sbatch
