#!/usr/bin/env python

# Randomly divide a folder of images into train (90%) and test (90%) sets.
# The train set will be split in code to ensure a 80/10/10 split.

import argparse
import os
import random
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing original images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output directory")
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="Test set fraction (default=0.1)")

    args = parser.parse_args()

    # error checking
    if args.test_fraction <= 0.0 or args.test_fraction >= 1.0:
        raise ValueError("Invalid test fraction: must be (0.0, 1.0)")

    if not os.path.isdir(args.input_dir):
        raise ValueError("Invalid input directory: ", args.input_dir)

    if os.path.isdir(args.output_dir):
        if not args.overwrite:
            raise ValueError("Output directory exists! (hint: --overwrite)")

        shutil.rmtree(args.output_dir)

    # copy each category separately
    for cat in next(os.walk(args.input_dir))[1]:
        # set up the directory structure
        for d in ("train", "test"):
            os.makedirs(os.path.join(args.output_dir, d, cat), exist_ok=True)

        all_images = os.listdir(os.path.join(os.path.join(args.input_dir, cat)))
        random.shuffle(all_images)

        # calculate the split
        split_index = round(len(all_images) * args.test_fraction)

        # copy the files (this is slow...)
        for n, f in enumerate(all_images[:split_index]):
            print("Copying cat ", cat, " test images [",
                  n+1, "/", split_index, "]\r",
                  sep="", end="", flush=True)
            srcfile = os.path.join(args.input_dir, cat, f)
            dstfile = os.path.join(args.output_dir, "test", cat, f)
            shutil.copyfile(srcfile, dstfile)
        print()

        for n, f in enumerate(all_images[split_index:]):
            print("Copying cat ", cat, " train images [", n+1, "/",
                  len(all_images) - split_index, "]\r",
                  sep="", end="", flush=True)
            srcfile = os.path.join(args.input_dir, cat, f)
            dstfile = os.path.join(args.output_dir, "train", cat, f)
            shutil.copyfile(srcfile, dstfile)
        print()

# EOF
