#!/usr/bin/env python

# hackety hack - tensorflow has lots of warnings. We don't want to see them.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from contextlib import redirect_stdout
from io import StringIO
from numpy import argmax
import sys
import tensorflow as tf

def accuracy_per_category(model, data_dir, verbose=False):
    # get the categories from the directory
    # this is pythonic bash magic
    categories = [os.path.split(d[0])[1] for d in os.walk(data_dir)][1:]
    categories.sort()

    image_size = model.input.shape[1:3]

    if verbose:
        print("image", *categories, "ground_truth", sep=",")

    cat_results = {}
    for cat in categories:
        cat_dir_path = os.path.join(data_dir, cat)
        num_images = len(os.listdir(cat_dir_path))
        cat_list = [int(cat)] * num_images

        # silence the summary info here
        with redirect_stdout(StringIO()):
            test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                cat_dir_path, image_size=image_size, labels=None)

        # FIXME memory leak with predict when run on a Dataset object?
        predictions = model.predict(test_ds, verbose=0)

        # FIXME this causes a larger leak :(
        # predictions = []
        # for d in test_ds:
            # predictions.append(model.predict(d, verbose=0)[0])

        correct = 0
        for idx, p in enumerate(predictions):
            preds = p

            # if len(p) == 1, we have a binary prediction.
            if len(p) == 1:
                preds = [1.0 - p[0], p[0]]

            predicted = argmax(preds)
            if predicted == int(cat):
                correct += 1

            if verbose:
                print(os.path.split(test_ds.file_paths[idx])[1], *preds, cat, sep=",")

        cat_results[int(cat)] = correct / len(predictions)
        tf.keras.backend.clear_session(free_memory=True)

    return cat_results

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", help="Model directory path", required=True)
    argparser.add_argument("--data", help="Test data directory path", required=True)
    argparser.add_argument("--cpu", help="Use CPU only (use when GPU is in use)", action="store_true")
    argparser.add_argument("--csv", help="Output in CSV format",  action="store_true")
    argparser.add_argument("--csv_prefix", help="data prepended to CSV output line (e.g. model name, ID, etc.)")
    argparser.add_argument("--verbose", action="store_true",
                           help="Output prediction data per image")
    args = argparser.parse_args()

    model_filename = args.model
    data_dir = args.data

    for d in [model_filename, data_dir]:
        if not os.path.exists(d):
            print("ERROR: directory does not exist:", d, file=sys.stderr)
            sys.exit(1)

    if args.cpu:
        # use CPU so we don't clobber our GPU training
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = tf.keras.models.load_model(model_filename)

    csv_line = None
    if args.csv and args.csv_prefix is not None:
        csv_line = args.csv_prefix

    for (cat, acc) in sorted(accuracy_per_category(model,
                                                   data_dir,
                                                   args.verbose).items()):
        if args.csv:
            if csv_line is not None:
                csv_line += ","
            else:
                csv_line = ""

            csv_line += str(cat) + "," + str(acc)
        elif not args.verbose:
            print("Category", cat, "Accuracy:", acc)

    if args.csv:
        print(csv_line)

# EOF
