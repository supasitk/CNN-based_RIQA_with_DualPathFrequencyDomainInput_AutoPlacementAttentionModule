#!/bin/bash
TF_VERSION=2.16.1
JOB_NAME="001_hyperparameter_search"
JOB_ID="j001"
CONTAINER_NAME="sk-RIQA-${JOB_NAME}-${JOB_ID}"
DOCKER_IMAGE="sk-tf:${TF_VERSION}-gpu-jupyter"
HOST_PATH="/raid/supasitk/"
CONTAINER_PATH="/sk"
WORKDIR_ROOT_NAME="test_RIQA"
CONTAINER_WORKDIR="${CONTAINER_PATH}/${WORKDIR_ROOT_NAME}/${JOB_NAME}/${JOB_ID}"
RUN_CMD="./run_train.sh"
SAVE_RESULT=true
HOST_EXEC_ROOT_DIR="${HOST_PATH}/${WORKDIR_ROOT_NAME}/${JOB_NAME}/${JOB_ID}"
HOST_EXEC_LOG_DIR="${HOST_EXEC_ROOT_DIR}/logs/train/*"
HOST_EXEC_CKPT_DIR="${HOST_EXEC_ROOT_DIR}/checkpoints/train/*"
RESULTDIR_ROOT_NAME="test_RIQA_result"
HOST_RESULT_ROOT_DIR="${HOST_PATH}/${RESULTDIR_ROOT_NAME}/${JOB_NAME}"
HOST_RESULT_LOG_DIR="${HOST_RESULT_ROOT_DIR}/log"
HOST_RESULT_CKPT_DIR="${HOST_RESULT_ROOT_DIR}/checkpoint"

# set SET_GPU_IDS_FROM_HERE=true if want to specify GPU ID devices from here then 
# environment variable "CUDA_VISIBLE_DEVICES" is set as given "GPU_IDS" variable.
# or
# set SET_GPU_IDS_FROM_HERE=false if environment variable "CUDA_VISIBLE_DEVICES" is already set
# from outside, e.g., user's environment variables on host/container, or scheduler (support SLURM). However, if "CUDA_VISIBLE_DEVICES" 
# is unavailable (not set or set to the empty string) from outside, then it is set as given "GPU_IDS" variable by default.
#
# for USE_SLURM=true, while Slurm configures each Slurm job with a cgroup that directs it to the correct GPUs, using Docker to run 
# another container escapes Slurm's confinement, and using --gpus=1 will always use the first GPU in the system, 
# whether that GPU is assigned to your job or not. When using Docker, you must consult the SLURM_JOB_GPUS (for sbatch) or 
# SLRUM_STEP_GPUS (for srun) environment variable and pass that along to your container. You should also impose limits on all other resources 
# used by your Docker container, so that your whole job stays within the resources allocated by Slurm's scheduler. 
USE_SLURM=true	
SET_GPU_IDS_FROM_HERE=false 
GPU_IDS=0
if [[ $SET_GPU_IDS_FROM_HERE == true ]]; then # SET_GPU_IDS_FROM_HERE=true (USE_SLURM should be false)
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "env CUDA_VISIBLE_DEVICES is set manually in the script as $CUDA_VISIBLE_DEVICES"
elif [[ $USE_SLURM == true ]]; then # USE_SLURM=true
    export CUDA_VISIBLE_DEVICES=$SLURM_STEP_GPUS
    echo "env CUDA_VISIBLE_DEVICES is set by Slurm scheduler with SLURM_STEP_GPUS environment variable as $CUDA_VISIBLE_DEVICES"
else # SET_GPU_IDS_FROM_HERE=false and USE_SLURM=false
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
# the corresponding GPU IDs given as same as host or allocation by the scheduler while the latter one make container seeing 
# only number of GPUs according to list of GPU IDs given and also always starting GPU IDs with 0 inside the container resulting in 
# the given GPU IDs in container may be # incorrect (not correspond to them on host or allocation by the scheduler). 

docker run \
-d \
-u $(id -u):$(id -g) \
--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
--gpus all \
--privileged \
--name $CONTAINER_NAME \
--rm -it \
-v $HOST_PATH:$CONTAINER_PATH \
$DOCKER_IMAGE \
"$@"
echo "Starting container was DONE."

# Exec container
docker exec \
--workdir ${CONTAINER_WORKDIR} \
$CONTAINER_NAME \
$RUN_CMD
echo "Executing job was DONE." 

# Stop container
docker stop $CONTAINER_NAME
echo "Stopping container was DONE."

# Copy log and checkpoint of execution to result directory
if $SAVE_RESULT; then
	if [ ! -d $HOST_RESULT_ROOT_DIR ]; then
		echo "Creating directory for saving result."
		mkdir -p $HOST_RESULT_LOG_DIR
		mkdir -p $HOST_RESULT_CKPT_DIR
	fi
	cp -r $HOST_EXEC_LOG_DIR $HOST_RESULT_LOG_DIR 
	cp -r $HOST_EXEC_CKPT_DIR $HOST_RESULT_CKPT_DIR
	echo "Saving execution result was DONE." 
fi

echo "END!!!"
