#!/bin/bash
TF_VERSION=2.16.1
HOST_PATH="/home/g6217550024"
CONTAINER_PATH="/sk"
SIF_IMAGE_PATH="$HOST_PATH/SIF_images/apptainer/TF-$TF_VERSION-gpu-jupyter.sif"
WORKDIR_ROOT_NAME="riqa_experiment"
JOB_NAME="set001"
JOB_ID="j001_first"
CONTAINER_WORKDIR="$CONTAINER_PATH/$WORKDIR_ROOT_NAME/$JOB_NAME/$JOB_ID"
RUN_CMD="./run_train.sh"
SAVE_RESULT=true
HOST_EXEC_ROOT_DIR="$HOST_PATH/$WORKDIR_ROOT_NAME/$JOB_NAME/$JOB_ID"
HOST_EXEC_LOG_DIR="$HOST_EXEC_ROOT_DIR/logs/train/*"
HOST_EXEC_CKPT_DIR="$HOST_EXEC_ROOT_DIR/checkpoints/train/*"
RESULTDIR_ROOT_NAME="$WORKDIR_ROOT_NAME/result"
HOST_RESULT_ROOT_DIR="$HOST_PATH/$RESULTDIR_ROOT_NAME/$JOB_NAME"
HOST_RESULT_LOG_DIR="$HOST_RESULT_ROOT_DIR/log"
HOST_RESULT_CKPT_DIR="$HOST_RESULT_ROOT_DIR/checkpoint"

# Load 'apptainer' or 'singularity-ce' module
#module load apptainer
module load singularity-ce

# Exec container (apptainer/singularity exec [exec options...] <container> <command>)
#apptainer exec \
singularity exec \
--nv \
--home $HOST_PATH:$CONTAINER_PATH \
--pwd $CONTAINER_WORKDIR \
$SIF_IMAGE_PATH \
$RUN_CMD
echo "Executing job was DONE."

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
