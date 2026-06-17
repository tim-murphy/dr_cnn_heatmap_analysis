#!/usr/bin/env bash

if [ $# -lt 1 ]
then
    echo "Usage: $0 <image_dir>"
    exit -1
fi

for m in 224 240 260 299 300 331 380 384 456 480 528 600
do
    ./resize.py --source_dir "$1" --output_dir "$1_$m" --width $m
done
