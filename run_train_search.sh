#!/bin/bash

# Top-level directory of project
PROJECT_HOME=`pwd`

# Add the top-level directory of project to the Python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME

# Set environment for matplotlib
export MPLCONFIGDIR=$PROJECT_HOME/mplconfigdir

# Set environment for CUDA and Tensorflow
export TF_FORCE_GPU_ALLOW_GROWTH=true

# set SET_GPU_IDS_FROM_HERE=true if want to specify GPU ID devices from here then 
# environment variable "CUDA_VISIBLE_DEVICES" is set as given "GPU_IDS" variable.
# or
# set SET_GPU_IDS_FROM_HERE=false if environment variable "CUDA_VISIBLE_DEVICES" is already set
# from outside, e.g., user's environment variables on host/container, or scheduler. However, if "CUDA_VISIBLE_DEVICES" 
# is unavailable (not set or set to the empty string) from outside, then it is set as given "GPU_IDS" variable by default.
SET_GPU_IDS_FROM_HERE=false 
GPU_IDS=0
if [[ $SET_GPU_IDS_FROM_HERE == true ]]; then # SET_GPU_IDS_FROM_HERE=true
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "env CUDA_VISIBLE_DEVICES is set manually in the script as $CUDA_VISIBLE_DEVICES"
else # SET_GPU_IDS_FROM_HERE=false
    if [[ ! -v CUDA_VISIBLE_DEVICES ]]; then 
        echo "env CUDA_VISIBLE_DEVICES is not set outside the script"
        export CUDA_VISIBLE_DEVICES=$GPU_IDS
        echo "env CUDA_VISIBLE_DEVICES need to be set manually with given GPU_IDS variable inside the script as $CUDA_VISIBLE_DEVICES"
    elif [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
        echo "env CUDA_VISIBLE_DEVICES is set to the empty string outside the script"
        export CUDA_VISIBLE_DEVICES=$GPU_IDS
        echo "env CUDA_VISIBLE_DEVICES need to be set manually with given GPU_IDS variable inside the script as $CUDA_VISIBLE_DEVICES"
    else
        echo "env CUDA_VISIBLE_DEVICES is already set outside the script as $CUDA_VISIBLE_DEVICES"
    fi
fi

# Arguments
CONFIG_PATH="./configs/train_search/riqa_search.yaml"
EXEC_PATH="train_search.py"

# Run 
python $EXEC_PATH --alsologtostderr --cfg_path=$CONFIG_PATH