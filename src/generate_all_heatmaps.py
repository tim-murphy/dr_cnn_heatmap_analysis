#!/usr/bin/env python

# [Re-]generate heatmaps for all models in a model_stats_*.csv file.
# Written by Tim Murphy <tim.murphy@canberra.edu.au> 2026

import argparse
import csv
import glob
import os

from train_thyself import generate_heatmap

if __name__ == '__main__':
    basedir = os.path.join(os.path.dirname(__file__), "..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_csv", type=str, required=True,
                        help="Trained model stats CSV file.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing graded images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--coordinates_csv", type=str,
                        default=os.path.join(basedir, "coordinates.csv"),
                        help="Path to mac/disc coordinates CSV file.")
    parser.add_argument("--grade_to_refer", action="store_true",
                        help="If set, will grade DR as referable or not.")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU for computations instead of GPU")
    parser.add_argument("--cmap", type=str, default="jet",
                        help="Colormap to use when generating the heatmaps")
    parser.add_argument("--exclude_img", nargs='+',
                        default=("007-0058-000", "007-0073-000", "007-1811-100",
                                 "007-1824-100", "007-2280-100", "007-2371-100",
                                 "007-2468-100", "007-2795-100"),
                        help="Exclude these image basenames")
    parser.add_argument("--model", nargs='+', default=[],
                        help="If set, only run these models")
    args=parser.parse_args()

    # Validate command line arguments.
    if not os.path.isfile(args.model_csv):
        raise ValueError("Invalid model_csv: " + args.model_csv)

    if not os.path.isdir(args.image_dir):
        raise ValueError("Invalid image_dir: " + args.image_dir)

    if not os.path.isfile(args.coordinates_csv):
        raise ValueError("Invalid coordinates_csv: " + args.coordinates_csv)

    with open(args.model_csv, 'r') as csvfile:
        for row in csv.DictReader(csvfile):
            model_path = os.path.join(os.path.dirname(__file__), row['model'])
            if not os.path.isfile(model_path):
                raise ValueError("Model does not exist: " + model_path)

            model_name = os.path.splitext(os.path.basename(model_path))[0]
            if len(args.model) > 0 and model_name not in args.model:
                print("Skipping model:", model_name)
                continue

            print("===", model_name, "===")

            generate_heatmap(model_name, model_path, args.output_dir,
                             args.image_dir, args.coordinates_csv,
                             cpu=args.cpu, cmap=args.cmap,
                             grade_to_refer=args.grade_to_refer,
                             exclude_img=args.exclude_img)

    print("All done! Have a nice day :)")

# EOF
