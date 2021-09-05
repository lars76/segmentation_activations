#!/usr/bin/env bash

export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="asia-east1-a"
export INSTANCE_NAME="my-instance"
export INSTANCE_TYPE="n1-highmem-2"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-k80,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True"