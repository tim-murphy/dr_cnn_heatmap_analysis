#!/usr/bin/env python

# from https://www.tensorflow.org/tutorials/images/cnn
# see tutorial https://www.tensorflow.org/tutorials/images/classification

# hackety hack - tensorflow has lots of warnings. We don't want to see them.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
import gc
from keras import backend as keras_backend
from keras.src.models.functional import Functional
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import sys

from model_common import loadModels
from models import get_model_config
from preprocessing import augment_dataset, scale_and_normalise
from test_model import accuracy_per_category

# prevent trying to allocate more memory on GPU than exists
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

# allow RAM/swap to be used to extend GPU VRAM
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from models import available_models, string_to_model

# Returns true if a model was trained, or False otherwise.
def train_model(args) -> bool:
    model_config = get_model_config(args.base_model)

    ### params ###
    colour_mode = args.colours
    num_colours = (1 if args.colours == "grayscale" else 3)
    img_width = int(model_config["image_width"])
    img_height = int(model_config["image_height"])
    model_filename = (None if args.retrain == "" else args.retrain)
    early_stopping = (not args.force_all_epochs)
    finetune = args.finetune

    ### error checking ###
    if not os.path.exists(args.data):
        print("ERROR: data directory does not exist:", args.data, file=sys.stderr)
        sys.exit(1)

    if finetune < 0.0 or finetune > 1.0:
        print("ERROR: finetune value must be in the range [0.0, 1.0]")
        sys.exit(1)

    data_dir = os.path.join(args.data, "train")
    test_dir = os.path.join(args.data, "test")

    for d in (data_dir, test_dir):
        if not os.path.isdir(d):
            raise ValueError("Data directory does not exist: " + d)

    for (val, name) in [(args.epochs, "epochs"), (args.batch_size, "batch_size"),
      (args.learning_rate, "learning_rate"), (args.dropout, "dropout"),
      (args.patience, "patience")]:
        if val < 0:
            print("ERROR: input", name, "should be positive, provided:", val, file=sys.stderr)
            sys.exit(1)

    if args.model_name == "":
        args.model_name = input("Model name: ")

    # have we already run this model?
    already_run = False
    if os.path.exists(args.stats_file):
         for m in loadModels(args.stats_file):
            if m.model == args.model_name:
                already_run = True
                break

    if already_run:
        print("Model already run (skipping):", args.model_name)
        return False

    ### build the model ###

    seed=111 # to keep the validation split the same
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(img_width, img_height),
        validation_split=0.111, # 80/10/10 but the test set is separate
        subset="training",
        seed=seed,
        color_mode=args.colours,
        batch_size=args.batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(img_width, img_height),
        validation_split=0.111,
        subset="validation",
        seed=seed,
        color_mode=args.colours,
        batch_size=args.batch_size)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    ### data augmentation ###
    train_ds = augment_dataset(train_ds)

    # verify the data - show some examples
    if (args.preview):
        plt.figure(figsize=(10, 15))
        for images, labels in train_ds.take(1):
            for i in range(num_classes):
                plt.subplot(2, 3, i+1)
                plt.imshow(images[i])
                plt.title(class_names[labels[i]])
                plt.axis("off")
                plt.grid(False)
        plt.show()
        sys.exit(0)

    input_shape = (img_width, img_height, (1 if args.redfree else num_colours))

    # BUG: efficientnet input shape is incorrect :(
    # https://github.com/keras-team/keras/issues/21529
    model_input = tf.keras.Input(shape=input_shape, name="input")
    preprocessing_model = model_input

    if args.redfree:
        def rgb_to_redfree(x):
            return x[:,:,:,1:2]

        preprocessing_model = tf.keras.layers.Lambda(rgb_to_redfree, name="redfree")(preprocessing_model)

    # are we re-training an existing model?
    model = None
    if model_filename is not None:
        # retraining
        if (not os.path.exists(model_filename)):
            print("Error: model does not exist: ", model_filename)
            sys.exit(1)

        model = tf.keras.models.load_model(model_filename)
    else:
        # not retraining
        base = string_to_model(args.base_model.lower())
        model = base(include_top=False, weights=model_config["weights"],
            input_tensor=None, pooling=None, classes=num_classes,
            input_shape=(input_shape),
            classifier_activation=model_config["activation"])

        model = tf.keras.Model(model_input, model(preprocessing_model))
        model.trainable = False

        image_batch, label_batch = next(iter(train_ds))
        feature_batch = model(image_batch)

        # add a classification head to the model
        averaging = tf.keras.layers.GlobalAveragePooling2D()
        feature_average = averaging(feature_batch)

        # If we only have two classes, make this a binary classifier.
        preds = None
        if num_classes == 2:
            preds = tf.keras.layers.Dense(1, activation="sigmoid")
        else:
            preds = tf.keras.layers.Dense(num_classes, activation="softmax")

        # Build the full model
        inputs = tf.keras.Input(shape=input_shape, name="image")
        x = model(inputs, training=False)
        x = averaging(x)
        x = tf.keras.layers.Dropout(args.dropout)(x)
        outputs = preds(x)
        model = tf.keras.Model(inputs, outputs)

    # If this is a binary classifier, use a different loss function.
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    if num_classes == 2:
        loss_function = "binary_crossentropy"

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=loss_function,
                  metrics=['accuracy'])
    model.build((None, *input_shape))

    if args.summary:
        model.summary(expand_nested=True)

    # stop training when we stop improving
    callbacks = []
    if early_stopping:
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience, # number of unsuccessful epochs before stopping
            restore_best_weights=True) # use the best results, not the last results
        ]

    ### train ###
    print("=== Training ===")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=[callbacks]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_epochs = len(history.history['accuracy'])

    keras_backend.clear_session()
    del history
    print("Garbage cleared:", gc.collect())

    ### finetune ###
    print("=== Finetuning ===")

    # we need to edit the parameters of the base model
    model_layer = None
    for layer in model.layers:
        if type(layer) != Functional:
            continue

        model_layer = layer.layers[1] # 0 is input, 1 is the actual model

    if model_layer is None:
        raise ValueError("Could not find base model layer!")

    model_layer.trainable = True
    finetune_layers = round(len(model_layer.layers) * args.finetune)

    for l in model_layer.layers[:finetune_layers]:
        l.trainable = False

    fine_patience = 2
    if early_stopping:
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=fine_patience, # number of unsuccessful epochs before stopping
            restore_best_weights=True) # use the best results, not the last results
        ]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate / 10.0),
                  loss=loss_function,
                  metrics=['accuracy'])

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=[callbacks]
    )

    # save our model for future use
    model.save(args.model_name)

    gc.collect()

    ### visualise ###

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    run_epochs = len(acc)
    epochs_range = range(run_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.ylim([0.0, 1.1])
    plt.plot([train_epochs-1, train_epochs-1], plt.ylim(), label="Fine Tuning")
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.ylim([0.0, max(loss + val_loss) * 1.1])
    plt.plot([train_epochs-1, train_epochs-1], plt.ylim(), label="Fine Tuning")
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    ### save stats ###

    # graph
    plt.savefig(args.model_name + "_training_graph.png")

    # graph data
    with open(args.model_name + "_training_graph_data.csv", 'w') as csvfile:
        print("acc", *acc, sep=",", file=csvfile)
        print("val_acc", *val_acc, sep=",", file=csvfile)
        print("loss", *loss, sep=",", file=csvfile)
        print("val_loss", *val_loss, sep=",", file=csvfile)
        print("epochs", train_epochs, len(history_fine.history['accuracy']),
              sep=",", file=csvfile)
        print("patience", args.patience, fine_patience, sep=",", file=csvfile)

    # stats
    if not os.path.exists(args.stats_file):
        with open(args.stats_file, 'w') as csvfile:
            acc_headers = []
            for i in range(num_classes):
                acc_headers.append("accuracy_" + str(i))

            print("model", *acc_headers, "epochs",
                  "batch_size", "learning_rate", "dropout",
                  "patience", "redfree",
                  sep=",", file=csvfile)

    cat_acc = accuracy_per_category(model, test_dir)

    with open(args.stats_file, 'a') as csvfile:
        print(args.model_name, *list(cat_acc.values()), run_epochs, args.batch_size, args.learning_rate,
              args.dropout, args.patience, args.redfree, sep=",", file=csvfile)

    print(args.model_name, "accuracy against test dataset:", *list(cat_acc.items()))

    # Clean up memory here or it will leak.
    tf.keras.backend.clear_session(free_memory=True)

    print("Finished!")
    return True

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", help="directory containing training and test data, split into train/test and category subdirectories", default=os.path.join("..", "refer"))
    argparser.add_argument("--retrain", help="if retraining a model, specify the model directory", default="")
    argparser.add_argument("--colours", help="colour mode", choices=["rgb", "grayscale"], default="rgb")
    argparser.add_argument("--epochs", help="maximum number of epochs", type=int, default=100)
    argparser.add_argument("--batch_size", help="number of images in each batch", type=int, default=16)
    argparser.add_argument("--learning_rate", help="CNN learning rate", type=float, default=0.0001)
    argparser.add_argument("--dropout", help="dropout rate (for supported models)", type=float, default=0.2)
    argparser.add_argument("--base_model", help="CNN to base model off", choices=available_models(), default="resnet50")
    argparser.add_argument("--model_name", help="model name, used as output directory name (prompt if not provided)", default="")
    argparser.add_argument("--preview", help="preview images from the dataset", action="store_true")
    argparser.add_argument("--summary", help="show model summary", action="store_true")
    argparser.add_argument("--force_all_epochs", help="continue training for the specified number of epochs even if no improvement", action="store_true")
    argparser.add_argument("--stats_file", help="CSV file to write training stats to", default="model_stats.csv")
    argparser.add_argument("--patience", help="number of worse epochs to run before stopping", type=int, default=2)
    argparser.add_argument("--weights", help="use these initial weights ('imagenet' or path to weights file)", type=str)
    argparser.add_argument("--redfree", help="convert images to be red-free (useful for retinal images)", action="store_true")
    argparser.add_argument("--finetune", type=float, default=0.1,
                           help="Finetune this portion of the model's layers (default 0.1)")
    argparser.add_argument("--cpu", action="store_true",
                           help="Use CPU for computations instead of GPU")
    args = argparser.parse_args()

    # cuda setup
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("WARN: no GPU present!")
        args.cpu = True

    if args.cpu:
        print("INFO: using CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with tf.device("/cpu:0" if args.cpu else "/gpu:0"):
        train_model(args)

# EOF
