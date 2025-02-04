#!/bin/bash
# usage: sbatch slurm_submit_apptainer_exec_RIQA.sh

#SBATCH --job-name=RIQA
#SBATCH --output=RIQA_%j.out
#SBATCH --error=RIQA_%j.err
#SBATCH --ntasks=1 
#SBATCH --qos=medium
#SBACTH --partition=gpuq
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH --gres=gpu:7g.80gb:1

module load slurm

srun bash apptainer_exec_RIQA.sh