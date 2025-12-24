#!/bin/bash
ApptainerDef="ApptainerDefinitionFile_for_TF2gpu.def"
SIF="TF-2.16.1-gpu-jupyter.sif"

module load apptainer
apptainer build $SIF
