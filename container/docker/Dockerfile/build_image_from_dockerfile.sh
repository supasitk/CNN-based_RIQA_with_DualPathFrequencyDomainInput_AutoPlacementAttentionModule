#!/bin/bash
TAG=sk-tf:2.16.1-gpu-jupyter
DOCKERFILE_DIR="/raid/supasitk/DNN/script/docker/Dockerfile"
DOCKERFILE_NAME="Dockerfile_for_TF2.gpu"

docker build -t $TAG -f $DOCKERFILE_NAME $DOCKERFILE_DIR
