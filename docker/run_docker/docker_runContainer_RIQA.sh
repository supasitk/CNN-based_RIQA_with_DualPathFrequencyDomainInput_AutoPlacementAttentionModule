#!/bin/bash
TF_VERSION=2.16.1

docker run \
-u $(id -u):$(id -g) \
--gpus all \
--privileged \
--name sk0-tf-$TF_VERSION-gpu-jupyter \
--rm -it \
-v /raid/supasitk/:/sk \
sk-tf:$TF_VERSION-gpu-jupyter \
"$@"
