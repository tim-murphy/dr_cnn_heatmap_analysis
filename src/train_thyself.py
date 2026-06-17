#!/usr/bin/env python

# note: need python 3.7 or newer for dataclass
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import gc
import glob
import os
import random
import shutil
import signal
import statistics
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only show ERROR and FATAL
import tensorflow as tf

from model_common import base_cnn, create_model_name, loadModels
from models import available_models, get_model_config
from train_model import train_model

 # heatmap utils
from create_heatmap import create_heatmap
from generate_graded_heatmaps import generate_graded_heatmaps

TOP_MODELS_TO_CONSIDER = 10
MODEL_TYPES = ("refer", "grade")
RAMDISK_PATH = "/mnt/ramdisk"

ALL_BASE_DIVISORS=(2.0,)

EXIT_FORCED = 112

@dataclass
class IntParameter:
    min_val: int
    max_val: int
    min_diff: int = 1

@dataclass
class FloatParameter:
    min_val: float
    max_val: float
    min_diff: float

PARAMETERS = {
        "batch_size": IntParameter(min_val=1, max_val=32, min_diff=1),
        "learning_rate": FloatParameter(min_val=0.000001, max_val=0.1, min_diff=0.000001),
        "dropout": FloatParameter(min_val=0.2, max_val=0.5, min_diff=0.1)
    }

@dataclass
class ModelArgs:
    data: str = None
    test_data: str = None
    retrain: str = ""
    colours: str = "rgb"
    epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 0.0001
    dropout: float = 0.2
    base_model: str = "resnet50"
    model_name: str = ""
    preview: bool = False
    height: int = -1
    summary: bool = False
    force_all_epochs: bool = False
    stats_file: str = None
    patience: int = 10
    redfree: bool = False
    model_type: str = "grade"
    finetune: float = 0.1
    heatmap_dir: str = None
    heatmap_img_dir: str = None
    coordinates_csv: str = None

    def data_dir(self):
        img_width = int(get_model_config(self.base_model)["image_width"])
        return self.model_type + "_" + str(img_width)

def generate_heatmap(model_name, model_path, heatmap_dir, heatmap_img_dir,
                     coordinates_csv, cpu, cmap, grade_to_refer=False,
                     exclude_img=[]):

    output_png = os.path.join(heatmap_dir, model_name + ".png")

    if os.path.exists(output_png):
        print("Heatmap already exists - skipping")
        return

    print("Generating heatmaps...")

    # note: this is created in generate_graded_heatmaps
    heatmap_tmp_dir = os.path.join(heatmap_dir, model_name)

    try:
        # generate individual heatmaps
        base_model = base_cnn(model_name)
        img_width = int(get_model_config(base_model)["image_width"])
        img_height = int(get_model_config(base_model)["image_height"])

        graded_args = lambda: None
        graded_args.model = model_path
        graded_args.datadir = heatmap_img_dir
        if heatmap_img_dir is None:
            graded_args.datadir = new_args.data_dir()
        graded_args.outdir = heatmap_tmp_dir
        graded_args.cpu = cpu
        graded_args.overwrite = True
        graded_args.threshold = 0
        graded_args.dims = (img_width, img_height)
        graded_args.grade_to_refer = grade_to_refer
        graded_args.exclude_img = exclude_img
        generate_graded_heatmaps(graded_args)

        # collate into one heatmap
        create_heatmap([
            "create_heatmap.py",
            coordinates_csv,
            heatmap_tmp_dir,
            heatmap_tmp_dir,
            "",
            cmap])

        # copy the collated heatmap and other associated files into the main
        # heatmap directory
        heatmap_files = [
            (os.path.join(heatmap_tmp_dir, "heatmap.png"), output_png)
        ]

        grade = -2 # -1 is used to mean "all grades".
        while (grade := (grade + 1)) < 100: # Better than while True / break.
            csvfile = os.path.join(
                heatmap_tmp_dir,
                "lesion_count_both_estimated_" + str(grade) + ".csv")

            if os.path.isfile(csvfile):
                heatmap_files.append((csvfile,
                                      output_png + "." + str(grade) + ".csv"))

        for (heatmap_file, output_file) in heatmap_files:
            if os.path.isfile(heatmap_file):
                shutil.copyfile(heatmap_file, output_file)
            else:
                print("ERROR: heatmap file does not exist: ",
                      heatmap_file, file=sys.stderr)

        # remove temporary files
        shutil.rmtree(heatmap_tmp_dir)

        print("Heatmap generation finished")
    except Exception as e:
        # something went wrong, usually a bad keras file
        print("ERROR:", e, file=sys.stderr)
        raise e

def train(args, model_name):
    basedir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = args.data_dir()
    src_data_dir = os.path.join(basedir, data_dir)
    dest_data_dir = getRamdiskPath(data_dir)

    if not os.path.isdir(src_data_dir):
        print("ERROR: image directory does not exist:", src_data_dir, file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(dest_data_dir):
        print("Copying", src_data_dir, "to", dest_data_dir, "... ", flush=True, end="")
        shutil.copytree(src_data_dir, dest_data_dir)
        print("done")

    args.data = dest_data_dir

    if args.stats_file is None:
        args.stats_file = os.path.join(basedir, "model_stats_" + args.model_type + ".csv")

    # check that all categories are numerical - will break later processing
    # if it's not
    args_bad = False
    for d in (os.path.join(args.data, "train"),
              os.path.join(args.data, "test")):
        cats = [f.path for f in os.scandir(d) if f.is_dir()]
        for cat in cats:
            try:
                int(os.path.split(cat)[-1])
            except:
                print("ERROR: category must be a number:", cat)
                args_bad = True

    if args_bad:
        sys.exit(1)

    print("")
    print(" ===", model_name, "===")
    print("")

    trained = False
    try:
        trained = train_model(args)
    except Exception as e:
        print("Error training model:", e, "(will try something else)")

    return trained

def getTrainedModelPath(model_type, model_name):
    # Make sure the folder to this path exists.
    basedir = os.path.join(os.path.dirname(__file__), "..")
    model_dir = os.path.join(basedir, "trained_" + model_type)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    return os.path.join(model_dir, model_name + ".keras")

def runAllBaseCNNs(args, model_type, base_models=available_models(),
                   cpu=False, single=True, cmap="jet", grade_to_refer=False):
    if base_models == [None] or len(base_models) == 0:
        base_models = available_models()

    trained = False
    for divisor in ALL_BASE_DIVISORS:
        if trained and single:
            break

        for base_nn in base_models:
            if trained and single:
                break

            batch_size = int(args.batch_size / divisor)
            learning_rate = args.learning_rate / divisor
            dropout = args.dropout / divisor

            # Special case to avoid OOM on my tiny GPU.
            if base_nn == "convnext_xlarge":
                batch_size = int(batch_size / 2)

            model_name = create_model_name(base_nn, batch_size,
                                           learning_rate, dropout)
            model_path = getTrainedModelPath(model_type, model_name)

            m_args = ModelArgs(base_model=base_nn,
                               batch_size=batch_size,
                               learning_rate=learning_rate,
                               dropout=dropout,
                               model_name=model_path,
                               model_type=model_type)

            with tf.device("/cpu:0" if cpu else "/gpu:0"):
                trained = train(m_args, model_name)

                if args.heatmap_dir is not None:
                    generate_heatmap(model_name, model_path, args.heatmap_dir,
                                     args.heatmap_img_dir, args.coordinates_csv,
                                     cpu=False, cmap=cmap,
                                     grade_to_refer=grade_to_refer)

def getRamdiskPath(directory):
    return os.path.join(RAMDISK_PATH, directory)

if __name__ == '__main__':
    basedir = os.path.join(os.path.dirname(__file__), "..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=MODEL_TYPES,
                        required=True, help="Type of AI model.")
    parser.add_argument("--single", action="store_true",
                        help="If set, will only create and train one model.")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU for computations instead of GPU")
    parser.add_argument("--base_model", help="CNN to base model off",
                        choices=available_models(), required=False)
    parser.add_argument("--heatmap_dir", help="If set, will generate heatmaps in this directory",
                        required=False, type=str, default=os.path.join(basedir, "heatmap_images"))
    parser.add_argument("--heatmap_img_dir", help="Directory contatining images used to generate heatmaps",
                        required=False, type=str)
    parser.add_argument("--coordinates_csv", help="Disc/mac coordinates file, used to generate heatmaps. Will use test directory if not set.",
                        default=os.path.join(basedir, "coordinates.csv"), type=str)
    parser.add_argument("--base_only", action="store_true",
                        help="Only train base CNNs")
    parser.add_argument("--cmap", help="CV2 colormap", default="jet", type=str)
    parser.add_argument("--grade_to_refer", action="store_true",
                        help="If set, will grade DR as referable or not.")
    args = parser.parse_args()

    # error checking
    if args.heatmap_dir is not None and not os.path.isdir(args.heatmap_dir):
        print("INFO: heatmap_dir does not exist. Creating...", end="")
        os.makedirs(args.heatmap_dir)
        print("done")

        # paranoia
        if not os.path.isdir(args.heatmap_dir):
            raise ValueError("Invalid heatmap dir: " + args.heatmap_dir)

    if args.heatmap_img_dir is not None and not os.path.isdir(args.heatmap_img_dir):
        raise ValueError("Invalid heatmap img dir: " + args.heatmap_img_dir)

    if args.heatmap_dir is not None and not os.path.isfile(args.coordinates_csv):
        raise ValueError("Invalid coordinates CSV file: " + args.coordinates_csv)

    model_type = args.model_type

    # copy the images to ramdisk to make things faster
    if not os.path.exists(RAMDISK_PATH):
        print("ERROR: ramdisk is not mounted at:", RAMDISK_PATH, file=sys.stderr)
        sys.exit(1)

    # cuda setup
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("WARN: no GPU present!")
        args.cpu = True

    if args.cpu:
        print("INFO: using CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # when this becomes false, we will quit
    keep_running = True
    force_exit = False

    # seed our randomiser
    random.seed(datetime.now().timestamp())

    def sigint_handler(sig, frame):
        global keep_running
        if not keep_running:
            # SIGINT twice - just exit now
            print("Ok, quitting right now. Goodbye.")
            sys.exit(EXIT_FORCED)

        keep_running = False
        force_exit = True
        print("Program will quit after this model finishes training.")
        print("Press Ctrl-C again to quit right now.")

    signal.signal(signal.SIGINT, sigint_handler)

    model_args = ModelArgs()
    model_args.heatmap_dir = args.heatmap_dir
    model_args.heatmap_img_dir = args.heatmap_img_dir
    model_args.coordinates_csv = args.coordinates_csv
    if model_args.stats_file is None:
        model_args.stats_file = os.path.join(basedir, "model_stats_" + model_type + ".csv")

    # run some default models here
    # FIXME duplication, doing almost the same thing as below.
    if not os.path.exists(model_args.stats_file):
        print("No models have been created yet. Running initial models.")
        runAllBaseCNNs(model_args, model_type, cpu=args.cpu,
                       single=args.single, cmap=args.cmap,
                       grade_to_refer=args.grade_to_refer)
        if args.single:
            keep_running = False

    # exit the program with this code
    exit_code = 0

    while keep_running:
        # load CSV data
        # note: we do this every time so we can have multiple processes
        model_stats = loadModels(model_args.stats_file, args.base_model)

        num_base_cnn_models = len(available_models()) * len(ALL_BASE_DIVISORS)
        heatmap_files = glob.glob(os.path.join(args.heatmap_dir, "*.png"))

        if len(model_stats) < num_base_cnn_models or\
        len(heatmap_files) < num_base_cnn_models:
            print("Running all base CNNs")
            runAllBaseCNNs(model_args, model_type, [args.base_model],
                           cpu=args.cpu, single=args.single, cmap=args.cmap)
            if args.single:
                keep_running = False
            continue

        # if we're only running base CNNs, we're done.
        if args.base_only:
            print("All base models run.")
            exit_code = EXIT_FORCED
            keep_running = False
            continue

        # order by overall accuracy
        model_stats.sort(reverse=True)
        top_models = model_stats[:TOP_MODELS_TO_CONSIDER]
        random.shuffle(top_models)
        top_models = top_models[:2]

        # parameters
        base_nn = [top_models[0].base(), top_models[1].base()]
        if base_nn[0] != base_nn[1]:
            random.shuffle(base_nn) # pick one at random
        base_nn = base_nn[0]

        # if we have specified a base CNN, use that instead
        if args.base_model is not None and args.base_model != "":
            base_nn = args.base_model

        params_changed = False

        # TODO: parameterise this somehow.
        batch_size = [top_models[0].batch_size, top_models[1].batch_size]
        if abs(batch_size[0] - batch_size[1]) <= PARAMETERS["batch_size"].min_diff:
            batch_size = batch_size[0]
        else:
            batch_size = int(statistics.mean(batch_size))
            params_changed = True

        learning_rate = [top_models[0].learning_rate, top_models[1].learning_rate]
        if abs(learning_rate[0] - learning_rate[1]) <= PARAMETERS["learning_rate"].min_diff:
            learning_rate = learning_rate[0]
        else:
            learning_rate = statistics.mean(learning_rate)
            params_changed = True

        dropout = [top_models[0].dropout, top_models[1].dropout]
        if abs(dropout[0] - dropout[1]) <= PARAMETERS["dropout"].min_diff:
            dropout = dropout[0]
        else:
            dropout = statistics.mean(dropout)
            params_changed = True

        if not params_changed:
            random_param, param_limits = random.choice(list(PARAMETERS.items()))
            new_value = random.randint(0, int((param_limits.max_val - param_limits.min_val) / param_limits.min_diff)) * param_limits.min_diff + param_limits.min_val

            if random_param == "batch_size":
                batch_size = new_value
            elif random_param == "learning_rate":
                learning_rate = new_value
            elif random_param == "dropout":
                dropout = new_value
            else:
                raise AttributeError("Randomised invalid parameter: " + random_param)

        model_name = create_model_name(base_nn, batch_size,
                                       learning_rate, dropout)
        model_path = getTrainedModelPath(model_type, model_name)

        # finally, see if this one has been done already
        do_training = True
        for model in model_stats:
            if model.base() == base_nn and\
               model.batch_size == batch_size and\
               model.dropout == dropout and\
               model.learning_rate == learning_rate:
                print(" ===", model_name, "===")
                print("New model has already been run. Selecting a new one.")
                do_training = False
                continue

            if model.model == model_path:
                print(" ===", model_name, "===")
                print("A model with the same name has already been run. Selecting a new one.")
                do_training = False
                continue

        if do_training:
            new_args = ModelArgs(base_model=base_nn,
                                 batch_size=batch_size,
                                 learning_rate=learning_rate,
                                 dropout=dropout,
                                 model_name=model_path,
                                 model_type=model_type)

            with tf.device("/cpu:0" if args.cpu else "/gpu:0"):
                train(new_args, model_name)

                # Explicitly run the garbage collector as we are often at the
                # limits of available memory, causing OOM errors when
                # iteratively training.
                gc.collect()

                # heatmaps
                if args.heatmap_dir is not None:
                    generate_heatmap(model_name, model_path, args.heatmap_dir,
                                     args.heatmap_img_dir, args.coordinates_csv,
                                     cpu=False, cmap=args.cmap,
                                     grade_to_refer=args.grade_to_refer)

            # Only run once if that's what we have set.
            if args.single:
                keep_running = False

    print("All done! Have a nice day :)")
    sys.exit(exit_code)

# EOF
