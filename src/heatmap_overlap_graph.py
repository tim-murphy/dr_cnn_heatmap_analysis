#!/usr/bin/env python

# Take output files from dice_lesion_stats.py and generate a graph from it.
# Written by Tim Murphy <tim.murphy@canberra.edu.au> 2026

import argparse
import csv
from enum import Enum
import matplotlib.pyplot as plt
import os
from statistics import mean, median, stdev

class DRFeature(str, Enum):
    MA = "Microaneurysms",
    HE = "Intraretinal haemorrhages",
    EX = "Exudates",
    SE = "Cotton wool spots",
    VB = "Venous beading",
    IRMA = "IRMA",
    NVD = "Neovascularistion at the disc",
    NVE = "Neovascularisation elsewhere"

class DRGrades(str, Enum):
    grade_0 = "No apparent DR",
    grade_1 = "Mild NPDR",
    grade_2 = "Moderate NPDR",
    grade_3 = "Severe NPDR",
    grade_4 = "Proliferative DR"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dice_stats_csv", type=str, nargs='+', required=True,
                        help="One or more stats CSV files created "\
                             "from dice_lesion_stats.py")
    parser.add_argument("--output_png", type=str, required=False,
                        help="File to write output graph")
    parser.add_argument("--error_bars", action="store_true",
                        help="Add error bars to the plot")
    parser.add_argument("--title", type=str,
                        default="Proportion of DR features covered by refer model heatmaps",
                        help="Plot title")
    parser.add_argument("--stat", type=str, choices=["mean", "median"],
                        default="median",
                        help="Stat to use as representative value")
    args = parser.parse_args()

    # Validate command line args.
    for f in args.dice_stats_csv:
        if not os.path.isfile(f):
            raise ValueError("Dice stats CSV file does not exist: " + str(f))

    # Format: dr_feature -> { grade -> list(float) }
    feature_data = {}
    for f in args.dice_stats_csv:
        grade_str = "<unknown>"
        try:
            grade_str = DRGrades[os.path.split(os.path.dirname(f))[1]]
        except Exception as e:
            print("WARN: could not convert DR grade to string:", e)

        with open(f, 'r') as csvfile:
            rows = csv.DictReader(csvfile)

            for row in rows:
                if row['model'] == "lesion" and row['feature'] == "n":
                    # We have reached the summary stats.
                    break

                if row[args.stat] == "N/A":
                    # No data for this feature for this grade.
                    continue

                dr_feature = DRFeature[row['feature']]

                if not dr_feature in feature_data:
                    feature_data[dr_feature] = {
                        DRGrades.grade_1: [],
                        DRGrades.grade_2: [],
                        DRGrades.grade_3: [],
                        DRGrades.grade_4: []
                    }

                feature_data[dr_feature][grade_str].append(float(row[args.stat]))

    all_grades = (
        DRGrades.grade_1,
        DRGrades.grade_2,
        DRGrades.grade_3,
        DRGrades.grade_4
    )

    plot_markers = (
        'o', 'v', '^', '>', '<', 's', 'd', 'P'
    )

    for i, (feature, grades) in enumerate(feature_data.items()):
        print("" + feature)

        plot_data = []
        plot_errors = [[], []]
        plot_grades = []
        for grade, means in grades.items():
            if len(means) == 0:
                continue

            # Median values.
            print("  " + grade, "::", round(median(means), 3),
                  "[", round(min(means), 3), "-", round(max(means), 3), "]")
            y = median(means)
            plot_errors[0].append(y - min(means))
            plot_errors[1].append(max(means) - y)

            # ...or mean values.
            if args.stat == "mean":
                print("  " + grade, "::", round(mean(means), 3), "SD =",
                      stdev(means))
                y = mean(means)
                plot_errors[0].append(y - stdev(means))
                plot_errors[1].append(y + stdev(means))

            x = all_grades.index(grade)
            plot_data.append(y)
            plot_grades.append(x)

        plt.errorbar(plot_grades, plot_data, label="" + feature,
                     marker=plot_markers[i],
                     yerr=(plot_errors if args.error_bars else None))

    fig = plt.gcf()
    fig.set_size_inches(10.5, 6)

    plt.ylim(0.0, 1.0)
    plt.xticks(range(0, len(all_grades)), ["" + g for g in all_grades])
    plt.legend(loc="lower left")
    plt.title(args.title)

    if args.output_png is None:
        plt.show()
    else:
        plt.savefig(args.output_png)
        print("Figure saved to", args.output_png)

    print()
    print("All done! Have a nice day :)")

# EOF
