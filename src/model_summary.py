#!/usr/bin/env python

# don't show tensorflow information messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from sys import argv, exit

model = None
if (len(argv) < 2):
    print("Usage: ", argv[0], " <model_path> <verbose>")
    exit(2)

model_filename = argv[1]
if (not os.path.exists(model_filename)):
    print("Error: model does not exist: ", model_filename)
    exit(1)

model = tf.keras.models.load_model(model_filename)

if len(argv) > 2:
    model.summary(expand_nested=True, show_trainable=True)
else:
    model.summary()

# EOF
