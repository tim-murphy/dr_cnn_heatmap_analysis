import tensorflow as tf
from tensorflow.keras import layers

from resnet import addResnetLayer

# FIXME params are not being passed in
def get_custom_model(input_shape=(224, 224, 3), num_classes=5, dropout_rate=0):
    # create a model
    kernel_size = (3, 3)
    strides = (1, 1)
    pool_size = (2, 2)
    activation = "softmax"

    # image input
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs
    x = layers.Conv2D(24, kernel_size, strides=strides, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=pool_size)(x)

    x = addResnetLayer(x, kernel_size, strides, pool_size, num_layers=2, filters=48)
    x = layers.MaxPooling2D(pool_size=pool_size)(x)
    x = addResnetLayer(x, kernel_size, strides, pool_size, num_layers=2, filters=96)
    x = layers.MaxPooling2D(pool_size=pool_size)(x)
    x = addResnetLayer(x, kernel_size, strides, pool_size, num_layers=2, filters=192)
    x = layers.MaxPooling2D(pool_size=pool_size)(x)

    # x = layers.Conv2D(16, kernel_size, strides=strides, padding='same', activation='relu')(x)
    # x = layers.Conv2D(32, kernel_size, strides=strides, padding='same', activation='relu')(x)
    # x = layers.Conv2D(64, kernel_size, strides=strides, padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=pool_size)(x)

    # x = addResnetLayer(x, kernel_size, strides, pool_size)
    # x = addResnetLayer(x, kernel_size, strides, pool_size)
    # x = addResnetLayer(x, kernel_size, strides, pool_size)
    # x = layers.MaxPooling2D(pool_size=pool_size)(x)

    # x = layers.Conv2D(16, kernel_size, strides=strides, padding='same', activation='relu')(x)
    # x = layers.Conv2D(32, kernel_size, strides=strides, padding='same', activation='relu')(x)
    # x = layers.Conv2D(64, kernel_size, strides=strides, padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=pool_size)(x)

    x = layers.Flatten()(x)

    # x = layers.Dense(1024, activation='relu')(x)
    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(num_classes, activation=activation)(x)

    # compile
    model = tf.keras.Model(inputs, x, name="ResNet")

    return model

# EOF
