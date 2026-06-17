#!/usr/bin/env python

# Convert model_stats_grade.csv (or similar) to use referable criteria instead.

import argparse
import csv
import glob
import os
import sys

if __name__ == '__main__':
    basedir = os.path.join(os.path.dirname(__file__), "..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_csv", type=str,
                        default=os.path.join(basedir, "model_stats_grade.csv"),
                        help="General stats for models grading DR")
    parser.add_argument("--output_csv", type=str,
                        default=os.path.join(basedir, "model_stats_grade_to_refer.csv"),
                        help="Output stats file")
    parser.add_argument("--accuracy_csv", nargs="+", type=str,
                        default=glob.glob(os.path.join(basedir, "accuracy_grade_to_refer", "*.csv")),
                        help="Accuracy csv files")
    args = parser.parse_args()

    # Validate command line args.
    if not os.path.isfile(args.stats_csv):
        raise ValueError("stats_csv file does not exist: " + args.stats_csv)

    for a in args.accuracy_csv:
        if not os.path.isfile(a):
            raise ValueError("accuracy_csv file does not exist: " + a)

    # Extract the stats from each accuracy file.
    new_stats = {}
    for a in args.accuracy_csv:
        cat_counts = [0, 0]
        cat_correct = [0, 0]
        with open(a, 'r') as csvfile:
            model = os.path.basename(a).replace(".csv", ".keras")
            for row in csv.DictReader(csvfile):
                cat = int(row["ground_truth"])
                confs = [float(row["0"]), float(row["1"])]
                pred = confs.index(max(confs))

                cat_counts[cat] += 1
                if pred == cat:
                    cat_correct[cat] += 1

        new_stats[model] = [cat_correct[0] / cat_counts[0],
                            cat_correct[1] / cat_counts[1]]

    # Generate the new file.
    outfile_rows = []
    tail_cols = 0 # The number of columns after the accuracy_n data.
    with open(args.stats_csv, 'r') as csvfile:
        for row in csv.DictReader(csvfile):
            # Header.
            if len(outfile_rows) == 0:
                # Keep all header fields except accuracy_[>1]. A bit dirty.
                header_fields = list(row.keys())[:3] +\
                    [i for i in list(row.keys())[3:] if "accuracy_" not in i]
                outfile_rows.append(header_fields)

                tail_cols = len(header_fields[3:])

            # Data rows.
            model = os.path.basename(row["model"])

            if not model in new_stats.keys():
                raise ValueError("No accuracy_csv for model " + model)

            model_row = [row["model"], *new_stats[model], *list(row.values())[-tail_cols:]]
            outfile_rows.append(model_row)

    # Write the new file.
    with open(args.output_csv, 'w') as ofile:
        for row in outfile_rows:
            print(*row, sep=",", file=ofile)

    print("All done! Have a nice day :)")

# EOF
