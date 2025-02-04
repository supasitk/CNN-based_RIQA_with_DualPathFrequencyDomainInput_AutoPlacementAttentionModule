#!/bin/bash
TF_VERSION=2.16.1

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

# Run container

# P.S. using [ --gpus all ] instead of [ --gpus \"device=${CUDA_VISIBLE_DEVICES}\" ] because container still see all GPUs and use
# the corresponding GPU IDs given as same as host while the latter one make container seeing only number of GPUs according to list of 
# GPU IDs given and also always starting GPU IDs with 0 inside the container resulting in the given GPU IDs in container may be 
# incorrect (not correspond to them on host). 

docker run \
-u $(id -u):$(id -g) \
--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
--gpus all \
--privileged \
--name sk-tf-$TF_VERSION-gpu-jupyter \
--rm -it \
-v /raid/supasitk/:/sk \
sk-tf:$TF_VERSION-gpu-jupyter \
"$@"
