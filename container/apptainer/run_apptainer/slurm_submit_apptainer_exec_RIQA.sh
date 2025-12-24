#!/bin/bash
# usage: sbatch slurm_submit_apptainer_exec_RIQA.sh

#SBATCH --job-name=RIQA
#SBATCH --output=RIQA_%j.out
#SBATCH --error=RIQA_%j.err
#SBATCH --ntasks=1 
#SBATCH --qos=medium
#SBATCH --partition=gpuq
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --nodelist=dgx-03
#SBATCH --account=g_g6217550024

module load slurm

srun bash apptainer_exec_RIQA.sh