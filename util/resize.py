#!/usr/bin/env python

from  argparse import ArgumentParser
import cv2
import glob
import os
import shutil

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Directory containing original images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write resized images.")
    parser.add_argument("--width", type=int, required=True,
                        help="Width of resized images, in pixels.")
    parser.add_argument("--height", type=int, required=False,
                        help="Height of resized images, in pixels. Will use --width value if not set.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite directory if it exists.")
    args=parser.parse_args()

    # Sanity checks.
    if not os.path.isdir(args.source_dir):
        raise ValueError("Source directory does not exist: " + args.source_dir)

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise ValueError("Output directory exists! Hint: use --overwrite.")
        else:
            print("INFO: overwriting output directory")
            shutil.rmtree(args.output_dir)

    if args.height is None:
        args.height = args.width

    if args.height <= 0 or args.width <= 0:
        raise ValueError("Width and height must be positive")

    # Setup.
    os.makedirs(args.output_dir)

    # Fetch a list of the files to resize.
    img = glob.glob(
        os.path.join("**", "*.*"),
        root_dir=args.source_dir,
        recursive=True)

    # Cache the created dirs to avoid checking existence for every file.
    newdirs = set()

    for n, i in enumerate(img):
        outdir = os.path.join(args.output_dir, os.path.split(i)[0])
        outfile = os.path.split(i)[1]

        if not outdir in newdirs:
            os.makedirs(outdir)
            newdirs.add(outdir)

        # Load and resize the image.
        resized = cv2.resize(
            cv2.imread(os.path.join(args.source_dir, i)),
            (args.width, args.height))
        cv2.imwrite(os.path.join(outdir, outfile), resized)

        print("\rResizing [", n+1, "/", len(img), "]",
              end="", sep="", flush=True)

    print()
    print("All done! Have a nice day :)")

# EOF

