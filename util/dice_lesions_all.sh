#!/usr/bin/env bash

DICE_LESIONS_PY="$(dirname $0)/../src/dice_lesions.py"
DICE_STATS_PY="$(dirname $0)/../src/dice_lesion_stats.py"

if [ $# -lt 2 ]
then
    echo "Usage: $0 <model_dir> <output_dir> [<image_dir> [<annotation_dir> [<threshold>]]]"
    exit 1
fi

model_dir="$1"
image_dir="$(dirname $0)/../heatmap_images"
annotation_dir="$(dirname $0)/../annotations"
threshold="otsu"

if [ $# -ge 3 ]
then
    image_dir="$3"
fi

if [ $# -ge 4 ]
then
    annotation_dir="$4"
fi

if [ $# -ge 5 ]
then
    threshold="$5"
fi

output_dir="$2/$threshold"

# Validate command line arguments.
if [ ! -d "$model_dir" ]
then
    echo "Model directory does not exist: $model_dir" >&2
    exit 1
fi

if [ ! -d "$image_dir" ]
then
    echo "Image directory does not exist: $image_dir" >&2
    exit 1
fi

if [ ! -d "$annotation_dir" ]
then
    echo "Annotation directory does not exist: $annotation_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"

thresh_args="--otsu"
if [ $threshold != "otsu" ]
then
    thresh_args="--proportion $threshold"
fi

models=($(find "$model_dir" -maxdepth 1 -name \*.keras))
declare -a threads
for m in ${models[@]}
do
    echo "=-=-= $(basename $m) =-=-="
    outfile_csv="$output_dir/$(basename $m).csv"

    if [ -e $outfile_csv ]
    then
        echo "File already exists (skipping): $outfile_csv"
    else
        ($DICE_LESIONS_PY --image_dir "$image_dir" --annotation_dir "$annotation_dir"\
            --model "$m" $thresh_args --outfile_csv "$outfile_csv" --cpu) &
        threads+=($!)
    fi
done

for t in "${threads[@]}"
do
    echo "Waiting for thread $t"
    wait $t
    echo "Thread $t finished"
done

# Collate all of the stats.
echo -n "Collating stats..."
$DICE_STATS_PY --dice_lesion_csv $output_dir/*.csv --output_csv "$output_dir/../dice_stats_$threshold.csv"
echo "done"

echo ""
echo "All done! Have a nice day :)"

# EOF
