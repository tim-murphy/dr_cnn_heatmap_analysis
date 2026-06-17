#!/usr/bin/env python

# calculate the Sorensen-Dice coefficient for two binary masks, as generated
# by create_threshold_mask.py (or similar).

import argparse
import csv
import cv2
import numpy as np
import os
from scipy.spatial import distance
import sys

from string_to_cmap import string_to_cmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap1_csv", type=str, required=True,
                        help="CSV file with heatmap data (matrix of intensity values)")
    parser.add_argument("--heatmap2_csv", type=str, required=True,
                        help="CSV file with heatmap data (matrix of intensity values)")
    parser.add_argument("--threshold1", type=int, default=0,
                        help="Intensity counts less than this value will be set to zero, the rest to 1")
    parser.add_argument("--threshold2", type=int, default=0,
                        help="Intensity counts less than this value will be set to zero, the rest to 1")
    parser.add_argument("--otsu", action="store_true",
                        help="If set, will apply adaptive thresholding via the Otsu method")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="If set, will store heatmaps and stats to this directory")
    parser.add_argument("--cmap", type=str, default="Jet",
                        choices=["autumn", "bone", "jet", "winter", "rainbow", "ocean", "summer", "spring", "cool", "hsv", "pink", "hot"],
                        help="Colormap to use for heatmap files")
    args = parser.parse_args()

    maskdata = []

    # Set up the output directory.
    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

    for maskfile, threshold in ((args.heatmap1_csv, args.threshold1),
                                (args.heatmap2_csv, args.threshold2)):
        if not os.path.exists(maskfile):
            print("ERROR: file does not exist:", maskfile, file=sys.stderr)
            sys.exit(1)

        with open(maskfile, 'r') as csvfile:
            # python magic: convert to bool via int
            thresh = np.array(list(csv.reader(csvfile, delimiter=',')), dtype=np.uint8)

            if args.otsu:
                _, thresh = cv2.threshold(thresh, 0, 255,
                                          cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
                threshold = 0

            thresh = np.where(thresh < threshold, 0, thresh)
            maskdata.append(thresh.astype(bool))

            # Generate heatmaps.
            if args.output_dir is not None:
                cv2.imwrite(os.path.join(args.output_dir,
                                         os.path.basename(maskfile) + ".heat.png"),
                            cv2.applyColorMap(thresh, string_to_cmap(args.cmap)))

    dice = 1 - distance.dice(*(m.flatten() for m in maskdata))

    print(dice)

# EOF
