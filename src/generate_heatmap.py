#!/usr/bin/env python

# https://keras.io/examples/vision/grad_cam/

# hackety hack - tensorflow has lots of warnings. We don't want to see them.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from keras.src.models.functional import Functional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
import sys

# Display
import matplotlib.pyplot as plt
import cv2

from csv_to_heatmap import csv_to_heatmap
from model_common import loadModels

# allow RAM/swap to be used to extend GPU VRAM
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 2
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

keras.config.disable_traceback_filtering()

def make_gradcam_heatmap(img_array, model):
    last_conv_layer = None
    layer_index = len(model.layers)
    for l0 in reversed(model.layers):
        # check that this is a Functional layer
        if type(l0) != Functional:
            layer_index -= 1
            continue

        for l in reversed(l0.layers):
            if l is None:
                continue

            for l2 in reversed(l._layers):
                if last_conv_layer == None and type(l2) == Conv2D:
                    last_conv_layer = l2.name
                    break

    if model.layers[1].layers[1].name in ("convnext_tiny", "convnext_small",
                                          "convnext_base", "convnext_large",
                                          "convnext_xlarge"):
        last_conv_layer = "layer_normalization"
    elif model.layers[1].layers[1].name == "xception":
        last_conv_layer = "block14_sepconv2_act"
    elif model.layers[1].layers[1].name == "nasnet_mobile":
        last_conv_layer = "normal_concat_12"
    elif model.layers[1].layers[1].name == "nasnet_large":
        last_conv_layer = "normal_concat_18"
    elif model.layers[1].layers[1].name in ("resnet50", "resnet101",
                                            "resnet152"):
        last_conv_layer = "conv5_block3_out"
    elif model.layers[1].layers[1].name in ("resnet50v2", "resnet101v2",
                                            "resnet152v2"):
        last_conv_layer = "post_relu"
    elif model.layers[1].layers[1].name == "densenet201":
        last_conv_layer = "conv5_block32_concat"
    elif model.layers[1].layers[1].name == "densenet169":
        last_conv_layer = "conv5_block32_2_conv"
    elif model.layers[1].layers[1].name == "densenet121":
        last_conv_layer = "conv5_block16_concat"

    if last_conv_layer is None:
        raise ValueError("Could not find last convolution layer!")

    # construct the gradient model
    grad_model = keras.models.Model(
        model.layers[layer_index].layers[-1].inputs,
        [
            model.layers[layer_index].layers[-1].get_layer(last_conv_layer).output,
            model.layers[layer_index].layers[-1].output
        ]
    )

    convOutputs = None
    # cast the image tensor to a float-32 data type, pass the
    # image through the gradient model, and grab the loss
    # associated with the specific class index
    inputs = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        (convOutputs, predictions) = grad_model(inputs)

        # pass the predictions through the rest of the layers
        for l in model.layers[layer_index+1:]:
            predictions = l(predictions)

        # In some instances, predictions are put in a list with one element.
        if type(predictions) == list:
            predictions = predictions[0]

        loss = predictions[:, tf.argmax(predictions[0])]

    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)

    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads

    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))

    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + 1e-8
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")

    # return the resulting heatmap to the calling function
    return heatmap

def create_heatmap_from_image(img_path, model, cmap_str="hot", preview=False, outdir=None, threshold=0, statsfile="grading.csv", outfiles=["original", "heatmap_coloured", "heatmap_grey", "overlay"], silent=False, dims=(224,224)):
    # TODO: skip if heatmaps have already been created.

    # prepare the image
    # note: target_size is (height,width) whereas dims is (width,height)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(dims[1], dims[0]))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # grade prediction
    preds = model.predict(img_array, verbose=0)[0]
    score = preds

    # special processing for binary classifiers
    if len(preds) == 1:
        score = [1.0 - preds[0], preds[0]]

    if not silent:
        print("The image {} most likely belongs to grade {} with a {:.2f}% confidence."
              .format(img_path, np.argmax(score), 100 * np.max(score)))

    # generate heatmap
    original = cv2.imread(img_path)
    heatmap = cv2.resize(make_gradcam_heatmap(img_array, model), original.shape[0:2][::-1])
    heatmap_coloured = csv_to_heatmap(heatmap, cmap_str, threshold,
                                      show_annotations = False)

    overlay = cv2.addWeighted(original, 0.5, heatmap_coloured, 0.5, gamma=0)

    if preview:
        # generate the output plot
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(15, 5)
        for i in range(3):
            ax[i].axis("off")
        ax[0].set_title("Original")
        ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Heatmap")
        ax[1].imshow(cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB))
        ax[2].set_title("Overlay")
        ax[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.show()

    # write to disk
    if outdir is not None:
        # create directory
        os.makedirs(outdir, exist_ok=True)
        filename = os.path.split(img_path)[1]

        # stats
        if statsfile is not None:
            csv_out = os.path.join(outdir, statsfile)
            if not os.path.exists(csv_out):
                # create the file with header row
                with open(csv_out, 'w', encoding='utf-8') as csvfile:
                    print("filename,grade,confidence", file=csvfile)

            with open(csv_out, 'a') as csvfile:
                print(filename, ",", np.argmax(score), ",", np.max(score),
                      sep='', file=csvfile)

        # write all heatmaps to this directory
        for (subdir, img) in [("original", original),
                              ("heatmap_coloured", heatmap_coloured),
                              ("heatmap_grey", heatmap),
                              ("overlay", overlay)]:
            if subdir not in outfiles:
                continue

            outpath = os.path.join(outdir, subdir, filename)

            # if we only have one type of outfile, don't put in a subdirectory
            if len(outfiles) == 1:
                outpath = os.path.join(outdir, filename)

            os.makedirs(os.path.split(outpath)[0], exist_ok=True)

            # the grey heatmap is in TIFF format, for later processing
            if subdir == "heatmap_grey":
                outpath = outpath[:-3] + "tif"

            cv2.imwrite(outpath, img)

    return np.argmax(score)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", help="model to use to generate heatmap data", required=True)
    argparser.add_argument("--outdir", help="output directory")
    argparser.add_argument("--preview", help="preview results", action="store_true")
    argparser.add_argument("--summary", help="show model summary", action="store_true")
    argparser.add_argument("--cmap", help="openCV colormap", default="hot", choices=["autumn", "bone", "jet", "winter", "rainbow", "ocean", "summer", "spring", "cool", "hsv", "pink", "hot"])
    argparser.add_argument("--threshold", help="heatmap threshold level (clear pixels lower than this value)", type=int, default=0)
    argparser.add_argument("--cpu", help="don't use GPU for computations", action="store_true")
    argparser.add_argument("--model_stats_csv", help="path to model stats CSV file", type=str, default=os.path.join("..", "model_stats_refer.csv"))
    argparser.add_argument("image_path", help="image or directory containing images", nargs="+", metavar="IMG")
    args=argparser.parse_args()

    # error checking
    args_valid = True
    if not os.path.exists(args.model):
        print("ERROR: model does not exist:", args.model, file=sys.stderr)
        args_valid = False

    if args.threshold < 0:
        print("ERROR: threshold value must be positive", file=sys.stderr)
        args_valid = False

    if not os.path.isfile(args.model_stats_csv):
        print("ERROR: model stats CSV file is not value:", args.model_stats_csv,
              file=sys.stderr)
        args_valid = False

    images = []
    for img in args.image_path:
        if os.path.isfile(img):
            images.append(img)
        elif os.path.isdir(img):
            for f in os.listdir(img):
                img2 = os.path.join(img, f)
                if os.path.isfile(img2):
                    images.append(img2)
                elif os.path.isdir(img2):
                    print("ignoring image subdirectory:", img2)
        else:
            print("ERROR: invalid image path:", img, file=sys.stderr)
            args_valid = False

    if len(images) == 0:
        print("ERROR: no images to process", file=sys.stderr)
        args_valid = False

    if args.cpu:
        # use CPU so we don't clobber our GPU training
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) == 0:
            print("WARN: no GPU present. Using CPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if not args_valid:
        sys.exit(1)

    # load model
    model = tf.keras.models.load_model(args.model)

    if args.summary:
        model.summary(expand_nested=True)

    # figure out the image dimensions from the model name
    img_dims = None
    for m in loadModels(args.model_stats_csv):
        if m.model == args.model:
            img_dims = m.img_dims()

    if img_dims is None:
        raise ValueError("Unknown model type: " + args.model +\
                         " (using stats file " + args.model_stats_csv + ")")

    for img in images:
        create_heatmap_from_image(img, model, args.cmap, args.preview, args.outdir, threshold=args.threshold, dims=img_dims)

# EOF
