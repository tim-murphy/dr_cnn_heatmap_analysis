#!/usr/bin/env python

# For a given model and set of images, run the image through the model,
# generate a GradCAM heatmap, threshold (dynamically) this heatmap and perfrom
# a Dice coefficient against each DR feature.

# hackety hack - tensorflow has lots of warnings. We don't want to see them.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import cv2
import numpy as np
from scipy.spatial import distance
import sys
import tempfile
import tensorflow as tf
from tensorflow import keras

from heatmap_threshold_graph import heatmap_threshold_graph
from generate_heatmap import make_gradcam_heatmap
import models

COMPARISONS=("dice", "pixels")

# Recursively extract image files from a directory.
def images_from_dir(dirname):
    images = []

    for f in os.listdir(dirname):
        img = os.path.join(dirname, f)
        if os.path.isfile(img):
            images.append(img)
        elif os.path.isdir(img):
            images += images_from_dir(img)

    return images

# Extract the folder names in a given directory. Non-recursive, relative path.
def folders_in_dir(dirname):
    folders = []

    for f in os.listdir(dirname):
        dirpath = os.path.join(dirname, f)
        if os.path.isdir(dirpath):
            folders.append(f)

    return folders

def dice_lesions(image_dir, annotation_dir, model, proportion, outfile_csv, comparison, otsu):
    if comparison not in COMPARISONS:
        raise ValueError("Invalid comparison: " + comparison)

    # Recursively extract the images.
    mod = tf.keras.models.load_model(model)
    images = images_from_dir(image_dir)

    # Extract the lesion types we will compare.
    lesions = folders_in_dir(annotation_dir)

    # Load the model config.
    model_conf = models.get_model_config(
        os.path.basename(model).split("~")[0])
    dims = (int(model_conf["image_width"]),
            int(model_conf["image_height"]))

    output_data = [["image", "prediction", *lesions]]

    for img_no, img in enumerate(images):
        if outfile_csv is None:
            print("--", img, "--")
        else:
            print("\r[", img_no + 1, "/", len(images), "]", end="")

        # prepare the image
        # note: target_size is (height,width) whereas dims is (width,height)
        # note: it would be more efficient to process the whole image
        #       directory in one hit, but keeping it like this as this is
        #       not run often.
        img_array = tf.keras.preprocessing.image.img_to_array(
                        tf.keras.preprocessing.image.load_img(
                            img, target_size=(dims[1], dims[0])))
        img_array = tf.expand_dims(img_array, 0)

        # Prediction.
        pred = mod.predict(img_array, verbose=0)
        if len(pred[0]) == 1:
            pred = (1 if pred[0] >= 0.5 else 0)
        else:
            pred = np.argmax(pred)

        # Generate heatmaps
        original = cv2.imread(img)
        heatmap = cv2.resize(make_gradcam_heatmap(img_array, mod), original.shape[0:2][::-1])
        threshold = int(proportion)

        # If we have set an absolute threshold, use that instead.
        if otsu:
            _, heatmap = cv2.threshold(heatmap, 0, 255,
                                       cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        else:
            if proportion <= 1:
                threshold = heatmap_threshold_graph(heatmap, proportion=proportion)

                # >254 will erase everything.
                if threshold > 254:
                    threshold = 254

            _, heatmap = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)

        # Dice each of the lesions.
        lesion_basename = os.path.splitext(os.path.join(os.path.basename(img)))[0] + ".tif"

        data_exists = False
        lesion_dice = {}
        for lesion in lesions:
            lesion_tif = os.path.join(annotation_dir, lesion, lesion_basename)

            # No data for this lesion.
            if not os.path.isfile(lesion_tif):
                lesion_dice[lesion] = None
                continue

            # Convert to binary. This is a binary file anyway.
            _, lesion_data = cv2.threshold(
                cv2.imread(lesion_tif, cv2.IMREAD_GRAYSCALE),
                127, 255, cv2.THRESH_BINARY)

            # If there is no lesion data for this image, we can ignore it.
            if np.max(lesion_data) == 0:
                lesion_dice[lesion] = None
                continue

            data_exists = True
            if comparison == "dice":
                dice = 1 - distance.dice(heatmap.astype(bool).flatten(),
                                         lesion_data.astype(bool).flatten())
                lesion_dice[lesion] = dice
            elif comparison == "pixels":
                lesion_indices = (lesion_data != 0)
                heatmap_indices = (heatmap != 0)
                intersect = np.logical_and(lesion_indices, heatmap_indices)
                num_pixels = np.count_nonzero(lesion_indices)
                lesion_dice[lesion] = (np.count_nonzero(intersect) / num_pixels)

        if data_exists:
            outdata = [os.path.basename(img), pred]
            for lesion in lesions:
                outdata.append(lesion_dice[lesion])
            output_data.append(outdata)
        else:
            print("ERROR: no data for", img, file=sys.stderr)

    # Print / write output data.
    ofile = sys.stdout
    if outfile_csv is not None:
        ofile = open(outfile_csv, 'w')

    for line in output_data:
        print(*line, sep=",", file=ofile)

    if outfile_csv is not None:
        ofile.close()
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing [folders of] images.")
    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="Directory contatining labelled folders of annotations.")
    parser.add_argument("--model", type=str, required=True,
                        help="Tensorflow model to use.")
    parser.add_argument("--proportion", type=float, default=0.1,
                        help="Threshold each heatmap to include this proportion of values. Default 0.1.")
    parser.add_argument("--cpu", help="don't use GPU for computations.",
                        action="store_true")
    parser.add_argument("--outfile_csv", type=str, required=False,
                        help="CSV file to write output. Note: will overwrite.")
    parser.add_argument("--comparison", choices=COMPARISONS, default="pixels",
                        help="Comparison type: Sorensen-Dice or number of lesion pixels covered (default).")
    parser.add_argument("--otsu", action="store_true",
                        help="Use Otsu thresholding")
    args = parser.parse_args()

    # Error checking.
    args_valid = True
    if not os.path.isdir(args.image_dir):
        print("ERROR: image directory does not exist:", args.image_dir,
              file=sys.stderr)
        args_valid = False

    if not os.path.isdir(args.annotation_dir):
        print("ERROR: annotation directory does not exist:",
              args.annotation_dir, file=sys.stderr)
        args_valid = False

    if not os.path.exists(args.model):
        print("ERROR: model does not exist:", args.model, file=sys.stderr)
        args_valid = False

    if args.proportion < 0.0 or args.proportion >= 255:
        print("ERROR: proportion must be in the range [0, 1] (proportional) or "
              "(1, 254] (absolute threshold)", file=sys.stderr)
        args_valid = False

    if not args_valid:
        sys.exit(1)

    # Configure tensorflow.
    if args.cpu:
        # use CPU so we don't clobber our GPU training
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) == 0:
            print("WARN: no GPU present. Using CPU", file=sys.stderr)
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dice_lesions(args.image_dir, args.annotation_dir, args.model,
                 args.proportion, args.outfile_csv, args.comparison,
                 args.otsu)

# EOF
