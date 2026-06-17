# ResNet implementation

import tensorflow as tf
from tensorflow.keras import layers, models

def addResnetLayer(inputs, kernel_size, strides, pool_size, num_layers=2, filters=16):
    if (num_layers < 2):
        raise ValueError("addResNetLayer num_layers must be >= 2")

    # outputs of each block
    output = []

    # first block
    output.insert(0, None)
    output[0] = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu')(inputs)

    for n in range(1, num_layers):
        # nth block
        output.insert(n, None)
        output[n] = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu')(output[n-1])
        output[n] = layers.add([output[n], output[n-1]])

    return layers.Activation('relu')(output[num_layers-1])

# eof
