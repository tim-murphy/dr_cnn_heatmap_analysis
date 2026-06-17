#!/usr/bin/env bash

show_usage()
{
    echo "Usage: $0 <trained_model_dir> <output_dir> <model_type> [<test_model.py flags>]"
}

create_data_dir()
{
    echo "/mnt/ramdisk/$1_$2/test"
}

get_data_dir()
{
    width=$($(dirname $0)/../src/model_summary.py "$1" 2>/dev/null | fgrep InputLayer | awk -F, '{ print $2 }' | sed 's/ //g')
    echo $(create_data_dir $2 $width)
}

if [ $# -lt 3 ];
then
    show_usage
    exit 1
fi

models_dir="$1"
output_dir="$2"
model_type="$3"
flags="$4"

# Validate command line arguments.
if [ ! -d "$models_dir" ];
then
    echo "ERROR: trained_model_dir does not exist: $models_dir" >&2
    exit 1
fi

if [ ! -d "$output_dir" ];
then
    echo -n "Output_dir does not exist. Creating..."
    mkdir -p "$output_dir"
    echo "done."
fi

# Fetch all of the trained models.
for model in $(find "$models_dir" -maxdepth 1 -name *.keras -exec basename {} .keras \;);
do
    echo " === $model ==="

    if [ -f "$output_dir/$model.csv" ]
    then
        echo "Model has already been tested - skipping."
    else
        data_dir=$(get_data_dir "$models_dir/$model.keras" "$model_type")
        $(dirname $0)/../src/test_model.py $flags --model "$models_dir/$model.keras" --data $data_dir --verbose > "$output_dir/$model.csv"
    fi
    echo ""
done

# EOF
