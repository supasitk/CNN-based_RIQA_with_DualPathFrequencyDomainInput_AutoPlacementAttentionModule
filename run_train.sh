#!/bin/bash

# Top-level directory of project
PROJECT_HOME=`pwd`

# Add the top-level directory of project to the Python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME

# Set environment for matplotlib
export MPLCONFIGDIR=$PROJECT_HOME/mplconfigdir

# Set environment for CUDA and Tensorflow
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=7

# Arguments
CONFIG_PATH="./configs/train/riqa.yaml"
EXEC_PATH="train.py"

# Run 
python $EXEC_PATH --alsologtostderr --cfg_path=$CONFIG_PATH