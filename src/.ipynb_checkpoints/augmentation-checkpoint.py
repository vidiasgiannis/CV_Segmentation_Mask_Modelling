import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from keras import callbacks

# Function to get all augmentation layers together
def augmentation_layers_geometric():
    return keras.Sequential([
        # Geometric augmentations
        layers.RandomFlip('horizontal'),
        layers.RandomTranslation(
            height_factor=0.1, 
            width_factor=0.1,
            fill_mode='constant'  # Important for mask integrity
        ),
    ])

def augmentation_layers_color():
    return keras.Sequential([
        # Color augmentations (enhanced from original)
        layers.RandomBrightness(factor=0.2, value_range=(0,1)),
        layers.RandomContrast(factor=0.1),
        # layers.RandomSaturation(factor=0.5),
        # layers.RandomHue(factor=0.1, value_range=(0,1)),
    ])

# functions for single augmentation
def random_flip():
    return layers.RandomFlip('horizontal')

def random_translation():
    return layers.RandomTranslation(
        height_factor=0.1, 
        width_factor=0.1,
        fill_mode='constant'  # Important for mask integrity
    )

def random_brightness():
    return layers.RandomBrightness(factor=0.2, value_range=(0,1))

def random_contrast():
    return layers.RandomContrast(factor=0.2)

def random_saturation():
    return layers.RandomSaturation(factor=0.2)

def random_hue():
    return layers.RandomHue(factor=0.1, value_range=(0,1))