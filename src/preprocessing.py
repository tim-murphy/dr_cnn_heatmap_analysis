# image data preprocessing
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.data.experimental import AUTOTUNE

# data preprocessing - common to train and validation
def scale_and_normalise(height, width):
    return tf.keras.Sequential([
        layers.Resizing(height, width),
        layers.Rescaling(1./255)
    ], name="preprocessing")

def augment_dataset(dataset):
    augmentations = [
        tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomContrast((0.5, 0.5)),
            layers.GaussianNoise(0.1)
        ]),
        tf.keras.Sequential([
            layers.RandomRotation(0.25),
            layers.RandomContrast((0.5, 0.5)),
            layers.GaussianNoise(0.1)
        ]),
        tf.keras.Sequential([
            layers.RandomZoom(0.4, 0.4),
            layers.RandomContrast((0.5, 0.5)),
            layers.GaussianNoise(0.1)
        ]),
        tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.GaussianNoise(0.1)
        ]),
        tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomZoom(0.3, 0.3),
            layers.RandomContrast((0.5, 0.5)),
            layers.GaussianNoise(0.1)
        ]),
        tf.keras.Sequential([
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.3, 0.3),
            layers.RandomContrast((0.5, 0.5)),
            layers.GaussianNoise(0.1)
        ]),
        tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.3, 0.3),
            layers.RandomContrast((0.5, 0.5)),
            layers.GaussianNoise(0.1)
        ])
    ]

    # take a copy of the original dataset passed in so that we don't grow the
    # dataset exponentially
    orig_dataset = dataset

    for img_aug in augmentations:
        dataset = dataset.concatenate(orig_dataset.map(lambda x, y: (img_aug(x), y), num_parallel_calls=AUTOTUNE))

    return dataset.shuffle(1000)

 # EOF
