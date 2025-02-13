import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from keras import callbacks

# Function to get all augmentation layers together
def get_augmentation_layers():
    return keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.1),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomHeight(factor=0.1),
        layers.RandomWidth(factor=0.1),
        # Add other augmentation techniques as needed
    ])

# functions for single augmentation
def random_flip():
    return layers.RandomFlip('horizontal')

def random_rotation():
    return layers.RandomRotation(factor=0.1) #layers.RandomRotation(factor=(-0.025, 0.025))
    

def random_translation():
    return layers.RandomTranslation(height_factor=0.1, width_factor=0.1)

def random_contrast():
    return layers.RandomContrast(factor=0.1)

def random_zoom():
    return layers.RandomZoom(height_factor=0.1, width_factor=0.1)

def random_height():
    return layers.RandomHeight(factor=0.1)

def random_width():
    return layers.RandomWidth(factor=0.1)