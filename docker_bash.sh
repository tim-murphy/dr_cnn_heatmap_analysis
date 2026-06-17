#!/usr/bin/env bash
docker run --rm --runtime=nvidia --gpus all --mount type=bind,src=/mnt/ramdisk,dst=/mnt/ramdisk --mount type=bind,src=/mnt/shared/source/repos/dr_cnn,dst=/cnn -it train_docker:latest bash
