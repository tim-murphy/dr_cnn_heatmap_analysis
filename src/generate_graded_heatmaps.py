#!/usr/bin/env python

# for a given model and test data, grade each image, generate a gradcam heatmap
# and write heatmaps to disk in two directories: actual and calculated grade.

import argparse
import glob
import os
import shutil
import sys
import tensorflow as tf

from generate_heatmap import create_heatmap_from_image

def generate_graded_heatmaps(args):
    ### error checking ###

    args_valid = True
    if not os.path.exists(args.model):
        print("ERROR: model does not exist:", args.model, file=sys.stderr)
        args_valid = False

    if not os.path.exists(args.datadir):
        print("ERROR: input data directory does not exist:", args.datadir, file=sys.stderr)
        args_valid = False

    if os.path.exists(args.outdir):
        if not args.overwrite:
            print("ERROR: output directory already exists:", args.outdir, file=sys.stderr)
            print("       use --overwrite to overwrite this directory", file=sys.stderr)
            args_valid = False
        else:
            print("WARN: overwriting output directory:", args.outdir)
            shutil.rmtree(args.outdir)

    if args.threshold < 0:
        print("ERROR: threshold value must be positive", file=sys.stderr)
        args_valid = False

    if not args_valid:
        sys.exit(1)

    if args.cpu:
        # use CPU so we don't clobber our GPU training
        print("INFO: using CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) == 0:
            print("WARN: no GPU present. Using CPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    os.makedirs(args.outdir)

    ### create each heatmap ###

    # load model
    model = tf.keras.models.load_model(args.model)

    grades = [os.path.split(x[0])[1] for x in os.walk(args.datadir)][1:]
    for grade in grades:
        all_images = glob.glob(os.path.join(args.datadir, grade, "*.jpg"))
        for (n, img) in enumerate(all_images):
            print("Copying grade ", grade, " [", n+1, "/", len(all_images),
                  "]\r", sep="", end="", flush=True)

            if os.path.splitext(os.path.basename(img))[0] in args.exclude_img:
                print("excluding", img)
                continue

            # create the heatmap and store in the actual grade folder
            outdir = os.path.join(args.outdir, "actual_" + grade)
            os.makedirs(outdir, exist_ok=True)
            estimated_grade = create_heatmap_from_image(img, model, outdir=outdir, threshold=args.threshold, outfiles=["heatmap_grey"], statsfile=None, silent=True, dims=args.dims)

            if args.grade_to_refer:
                estimated_grade = (0 if estimated_grade <= 1 else 1)

            # copy the file to the estimated directory
            outdir2 = os.path.join(args.outdir, "estimated_" + str(estimated_grade))
            os.makedirs(outdir2, exist_ok=True)
            outfile = os.path.split(img)[1][:-3] + "tif"
            shutil.copyfile(os.path.join(outdir, outfile), os.path.join(outdir2, outfile))
        print()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", help="model to use to generate heatmaps", required=True)
    argparser.add_argument("--datadir", help="directory with input data, split by category into subdirectories", required=True)
    argparser.add_argument("--outdir", help="output directory", required=True)
    argparser.add_argument("--overwrite", help="overwrite output directory", action="store_true")
    argparser.add_argument("--threshold", help="heatmap threshold level (clear pixels lower than this value)", type=int, default=0)
    argparser.add_argument("--cpu", help="don't use GPU for computations", action="store_true")
    argparser.add_argument("--width", help="scale image to this width", default=224, type=int)
    argparser.add_argument("--height", help="scale image to this height", default=224, type=int)
    argparser.add_argument("--grade_to_refer", action="store_true", help="If set, will grade DR as referable or not")
    argparser.add_argument("--exclude_img", type=str, nargs="*", default=[],
                           help="Images to exclude")
    args = argparser.parse_args()
    args.dims = (args.width, args.height)

    generate_graded_heatmaps(args)

# EOF
