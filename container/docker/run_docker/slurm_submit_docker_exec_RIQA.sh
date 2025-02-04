#!/bin/bash
# usage: sbatch slurm_submit_docker_exec_RIQA.sh

#SBATCH --job-name=RIQA
#SBATCH --output=RIQA_%j.out
#SBATCH --error=RIQA_%j.err
#SBATCH --ntasks=1 
#SBATCH --gpus-per-task=1 
#SBATCH --gpu-bind=closest 

srun bash docker_exec_RIQA.sh
