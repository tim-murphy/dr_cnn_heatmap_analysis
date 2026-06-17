#!/usr/bin/env bash

ENSEMBLE_PY="$(dirname $0)/../src/ensemble.py"
OUTPUT_DIR="$(dirname $0)/../accuracy_grade_to_refer"

if [ $# -lt 1 ]
then
    echo "Usage: $0 <accuracy_csv> [<accuracy_csv> [...]]"
fi

mkdir -p "$OUTPUT_DIR"

for f in $@
do
    $ENSEMBLE_PY --accuracy_csv "$OUTPUT_DIR/$(basename "$f")" --csv_files "$f" --grade_to_refer
done
