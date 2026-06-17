#!/usr/bin/env python

# Take a heatmap CSV file with raw lesion counts and create CSV file showing
# how many elements have a value greater than X for all X in matrix. Then
# turn this into a threshold graph.

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def heatmap_threshold_graph(heatmap_data, output_csv=None, output_png=None, proportion=0.5):
    # data labels - used in output files and graphs
    LABEL_THRESH = 'Threshold'
    LABEL_COUNT = 'Count'
    LABEL_PROP = 'Proportion' # of elements above the threshold

    # this is not the most efficient data structure but makes graphing easy
    lesion_counts = {LABEL_THRESH: [],
                     LABEL_COUNT: [],
                     LABEL_PROP: []}

    # calculate the lesion counts for a given threshold
    lesion_count_max = (heatmap_data > 0).sum()
    for thresh in range(np.max(heatmap_data)):
        lesion_count = (heatmap_data > thresh).sum()
        lesion_counts[LABEL_THRESH].append(thresh)
        lesion_counts[LABEL_COUNT].append(lesion_count)
        lesion_counts[LABEL_PROP].append(lesion_count / lesion_count_max)

    # write to file
    if output_csv is not None:
        with open(output_csv, 'w') as csvfile:
            print(','.join(list(lesion_counts.keys())), file=csvfile)
            for i, thresh in enumerate(lesion_counts[LABEL_THRESH]):
                print(thresh, lesion_counts[LABEL_COUNT][i], lesion_counts[LABEL_PROP][i],
                      sep=',', file=csvfile)

    # create graph
    if output_png is not None:
        x_data = LABEL_THRESH
        y_data = LABEL_PROP
        plt.plot(lesion_counts[x_data], lesion_counts[y_data])
        plt.xlabel(x_data)
        plt.ylabel(y_data)
        plt.title("Proportion of locations with lesion counts above a set threshold")
        plt.savefig(output_png, dpi=300, bbox_inches='tight')

    # Find the threshold which is closest to 50%
    # This is not elegant :-(
    thresh = 0
    closest_prop = -1
    for idx, prop in enumerate(lesion_counts[LABEL_PROP]):
        if abs(prop - proportion) < abs(closest_prop - proportion):
            closest_prop = prop
            thresh = lesion_counts[LABEL_THRESH][idx]

    return thresh

def printUsage():
    print("Usage:", __file__, "<heatmap_csv> [<output_csv> [<output_png>]]")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        printUsage()
        sys.exit(1)

    heatmap_csv = sys.argv[1]
    if not os.path.exists(heatmap_csv):
        print("ERROR: heatmap_csv does not exist:", heatmap_csv, file=sys.stderr)
        sys.exit(1)

    output_csv = (None if len(sys.argv) <= 2 else sys.argv[2])
    output_png = (None if len(sys.argv) <= 3 else sys.argv[3])

    # load the CSV data
    raw_data = None
    with open(heatmap_csv, 'r') as csvfile:
        raw_data = np.array(list(csv.reader(csvfile, delimiter=',')), dtype=np.uint32)

    thresh = heatmap_threshold_graph(raw_data, output_csv, output_png)

    if output_csv is None and output_png is None:
        print(thresh)

# EOF
