#!/bin/bash
TF_VERSION=2.16.1

docker run \
-u $(id -u):$(id -g) \
--gpus all \
--privileged \
--name sk-tfboard-$TF_VERSION-gpu-jupyter \
--rm -it \
-v /raid/supasitk/:/sk \
-p 8066:6006 \
sk-tf:$TF_VERSION-gpu-jupyter \
"$@"

################################ How to Run and Access ######################################### 
# If you want to access the Tensorboard through the IP address or domain name of the server,   #
# add --host 0.0.0.0 to tensorboard command. 						       #	
#											       #	
#	tensorboard --logdir path_to_logs --host 0.0.0.0				       #
#											       #		
# You can access the Tensorboard page with http://<host>:8066 on any kind of machine connected #
# to the internet.									       #	
################################################################################################
