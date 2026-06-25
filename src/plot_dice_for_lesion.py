#!/usr/bin/env python

# Take results from util/dice_everything.sh and generate a plot to show stats
# across all models.
# Written by Tim Murphy <tim.murphy@canberra.edu.au> 2026

import argparse
import csv
import matplotlib.pyplot as plt
import os
from statistics import median

from models import string_to_model

FEATURES = {
    "MA": "Microaneurysms",
    "HE": "Haemorrhages",
    "EX": "Exudates",
    "SE": "Cotton Wool Spots",
    "VB": "Venous Beading",
    "NVD": "Neovascularisation at the Disc",
    "NVE": "Neovascularisation Elsewhere",
    "IRMA": "Intraretinal Microvascular Abnormalities"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dice_stats_csv", nargs="+", required=True,
                        help="dice_stats_otsu.csv (or similar)")
    parser.add_argument("--lesion", type=str, required=True,
                        choices=FEATURES.keys(), help="Lesion to plot")
    parser.add_argument("--title", type=str,
                        default="Proportion of %%FEATURE%% Covered by Heatmaps"\
                                " (median with range)")
    parser.add_argument("--errorbars", action="store_true",
                        help="Show error bars on plot")
    parser.add_argument("--boxplot", action="store_true",
                        help="Generate a boxplot")
    parser.add_argument("--output_png", type=str, required=False,
                        help="If set, will write plot to disc at this location")
    args = parser.parse_args()

    # Validate command line arguments.
    for d in args.dice_stats_csv:
        if not os.path.isfile(d):
            raise ValueError("dice_stats_csv does not exist: " + d)

    # Extract the data.
    # Format: model_name -> [raw_data]
    models = {}
    for dice_file in args.dice_stats_csv:
        with open(dice_file, 'r') as csvfile:
            rows = csv.DictReader(csvfile)

            if args.boxplot:
                box_data = []
                for row in rows:
                    lesion = row['feature']
                    if not lesion == args.lesion:
                        continue

                    # If there is no raw data then we can skip this row.
                    if not None in row:
                        continue

                    model = string_to_model(row['model']).__name__

                    if not model in models:
                        models[model] = []

                    models[model] += [ float(v) for v in row[None] ]
            else:
                for i, row in enumerate(rows):
                    lesion = row['feature']
                    if not lesion == args.lesion:
                        continue

                    # If there is no raw data then we can skip this row.
                    if not None in row:
                        continue

                    model = string_to_model(row['model']).__name__

                    if not model in models:
                        models[model] = []

                    models[model] += [ float(v) for v in row[None] ]

    # Generate the plots.
    if args.boxplot:
        plt.boxplot(models.values(),
                    medianprops={"color": "black", "linewidth": 2},
                    boxprops={"facecolor": "lightblue", "color": "black"},
                    patch_artist=True)
    else:
        for n, raw_data in enumerate(models.values()):
            med = median(raw_data)

            yerr = [[med - min(raw_data)], [max(raw_data) - med]]
            if not args.errorbars:
                yerr = None

            plt.errorbar(n,
                         med, marker='o',
                         yerr=yerr)



    xtick_offset = 1 if args.boxplot else 0
    plt.xticks(range(xtick_offset, len(models.keys()) + xtick_offset),
               models.keys(), rotation=90)
    plt.title(args.title.replace("%%FEATURE%%", FEATURES[args.lesion]))
    plt.gcf().set_figwidth(10) # inches
    plt.tight_layout()

    if args.output_png is None:
        plt.show()
    else:
        plt.savefig(args.output_png)
        print("Plot written to", args.output_png)

    print()
    print("All done! Have a nice day :)")

# EOF
