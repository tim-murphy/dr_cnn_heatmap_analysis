#!/usr/bin/env python

# Generate a heatmap from heatmap CSV data.
# Written by Tim Murphy <tim.murphy@canberra.edu.au> 2026.

import argparse
import cv2
import numpy as np
import os

from models import string_to_model
from string_to_cmap import string_to_cmap

# Default coordinates of the optic nerve and macula on the heatmap canvas.
NERVE_COORDS = (700, 550)
MAC_COORDS = (450, 575)

# Generate a heatmap from CSV data.
def csv_to_heatmap(heatmap_csv_data: np.typing.NDArray[np.uint8],
                   cmap: str,
                   threshold: int = 0,
                   show_annotations: bool = True,
                   annotation_colour_bgr: (int, int, int) = (255, 255, 255),
                   nerve_coords: (int, int) = NERVE_COORDS,
                   macula_coords: (int, int) = MAC_COORDS,
                   label_text: str = None,
                   label_coords: (int, int) = (30, 60),
                   label_scale: float = 1.5,
                   label_thickness: int = 2):

    # Normalise the data to be in the range [0, 255].
    if np.max(heatmap_csv_data) > 0:
        heatmap_csv_data *= int(255 / np.max(heatmap_csv_data))

    # Threshold the heatmap to remove noise.
    if threshold == "otsu":
        _, heatmap_csv_data = cv2.threshold(heatmap_csv_data, 0, 255,
                                            cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    elif threshold > 0:
        _, heatmap_csv_data = cv2.threshold(heatmap_csv_data, threshold, 255,
                                            cv2.THRESH_TOZERO)

    heatmap = cv2.applyColorMap(heatmap_csv_data, string_to_cmap(cmap))

    if show_annotations:
        # Add the optic nerve visualisation.
        cv2.circle(heatmap, nerve_coords, 45, annotation_colour_bgr, 2)
        cv2.circle(heatmap, nerve_coords, 30, annotation_colour_bgr, 2)
        cv2.circle(heatmap, nerve_coords, 15, annotation_colour_bgr, 2)

        # Add the macula annotation.
        cv2.circle(heatmap, macula_coords, 25, annotation_colour_bgr, 2)

    if label_text is not None:
        cv2.putText(heatmap, label_text, label_coords, cv2.FONT_HERSHEY_DUPLEX,
                    label_scale, annotation_colour_bgr, label_thickness,
                    cv2.LINE_AA)

    return heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap_csv", type=str, nargs="+", required=True,
                        help="Path to CSV file(s) containing heatmap data.")
    parser.add_argument("--cmap", type=str, default="jet",
                        help="OpenCV colourmap.")
    parser.add_argument("--threshold", type=str, default=0,
                        help="Only include CSV data greater than this value, or \"otsu\".")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="If set, will save heatmap image to this directory.")
    parser.add_argument("--no_annotations", action="store_true",
                        help="If set, will not add disc/macula annotations.")
    parser.add_argument("--label_text", type=str, required=False,
                        help="If set, will add text to the heatmap.")
    parser.add_argument("--infer_label", action="store_true",
                        help="If set, will use the model name as label_text.")
    parser.add_argument("--combined_output_png", type=str, required=False,
                        help="If set, will generate a combined heatmap.")
    args = parser.parse_args()

    # Validate command line arguments.
    for csvfile in args.heatmap_csv:
        if not os.path.isfile(csvfile):
            raise ValueError("Invalid heatmap CSV file: " + csvfile)

    thresh = args.threshold
    if args.threshold != "otsu":
        thresh = int(args.threshold)

        if thresh < 0 or thresh > 255:
            raise ValueError("Threshold must be between 0 and 255 (inclusive)")

    # Note: cmap is validated in the csv_to_heatmap function.

    heatmap_csvs = args.heatmap_csv.copy()
    if args.combined_output_png:
        heatmap_csvs.append(args.combined_output_png)

    for csvfile in heatmap_csvs:
        # Extract the model name, if applicable.
        label_text = args.label_text

        if csvfile == args.combined_output_png:
            if args.label_text is not None or args.infer_label == True:
                label_text = "Combined"
        elif args.infer_label:
            label_text = string_to_model(
                os.path.basename(csvfile).split("~")[0]).__name__

        # Generate the heatmap and write to file or display in popup window.
        heatmap_data = None

        if csvfile == args.combined_output_png:
            # Combine al of the heatmaps together.
            for c in args.heatmap_csv:
                h = np.loadtxt(c, delimiter=',').astype("uint32")
                if heatmap_data is None:
                    heatmap_data = h
                else:
                    heatmap_data += h

            # Normalise to uint8.
            heatmap_data = np.array([((255 / np.max(heatmap_data)) * h) for h in heatmap_data]).astype("uint8")
        else:
            heatmap_data = np.loadtxt(csvfile, delimiter=',').astype("uint8")

        heatmap = csv_to_heatmap(heatmap_data, args.cmap, thresh,
                                 show_annotations = (not args.no_annotations),
                                 label_text = label_text)

        if args.output_dir is None:
            cv2.imshow("Heatmap: " + csvfile, heatmap)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)

            output_png = os.path.join(args.output_dir,
                                      os.path.basename(csvfile) + ".png")

            if csvfile == args.combined_output_png:
                output_png = os.path.join(args.output_dir,
                                          args.combined_output_png)

            cv2.imwrite(output_png, heatmap)
            print("Heatmap written to", output_png)

    print("All done! Have a nice day :)")

# EOF
