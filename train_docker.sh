#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_type> [<base_model> [<flags>]"
    exit 1
fi

if ! docker build -t train_docker:latest .; then
    echo "ERROR: could not build docker container!"
    exit 1
fi

base_model_args=""
if [ $# -ge 2 -a "$2" != "" ]; then
    base_model_args="--base_model $2"
fi

other_flags=""
if [ $# -ge 3 ]; then
    other_flags="$3"
fi

# Run this as a loop as the tensorflow code has a small memory leak.
# 112 is a magic number from train_thyself when exiting with ctrl-c or any
# other condition where we want to stop this loop.
ret=0
while [ $ret -ne 112 ]; do
    docker run --rm --runtime=nvidia --gpus all --mount type=bind,src=/mnt/ramdisk,dst=/mnt/ramdisk --mount type=bind,src=/mnt/shared/source/repos/dr_cnn,dst=/cnn -it train_docker:latest python train_thyself.py --model_type $1 $base_model_args --heatmap_dir ../heatmaps_$1 --heatmap_img_dir ../$1_heatmap_images $other_flags --single;
    ret=$?
done

echo "All done! Have a nice day :)"
