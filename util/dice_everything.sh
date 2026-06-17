#!/usr/bin/env bash

RAMDISK=/mnt/ramdisk

if [ ! -d $RAMDISK ]
then
    echo "ERROR: ramdisk does not exist at $RAMDISK !" >&2
    exit 1
fi

if [ $# -lt 1 ]
then
    echo "Usage: $0 <grade|refer>"
    exit 1
fi

model_type=$1

if [ ! -d $RAMDISK/heatmap_images ]
then
    echo -n "Copying heatmap images to ramdisk..."
    cp -r $(dirname $0)/../heatmap_images $RAMDISK/
    echo "done"
fi

if [ ! -d $RAMDISK/annotations ]
then
    echo -n "Copying annotations to ramdisk..."
    cp -r $(dirname $0)/../annotations $RAMDISK/
    echo "done"
fi

for g in 4 3 2 1
do
    echo "=== Grade $g ==="
    $(dirname $0)/dice_lesions_all.sh $(dirname $0)/../trained_$model_type $(dirname $0)/../dice_lesions_$model_type/grade_$g $RAMDISK/heatmap_images/$g $RAMDISK/annotations otsu
    echo
done

echo "All done! Have a nice day :)"
