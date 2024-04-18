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

# Run container
docker run \
-d \
-u $(id -u):$(id -g) \
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