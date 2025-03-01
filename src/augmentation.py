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
        layers.RandomRotation(factor=0.02),
    ])

def augmentation_layers_color():
    return keras.Sequential([
        # Color augmentations (enhanced from original)
        layers.RandomBrightness(factor=0.2, value_range=(0,1)),
        layers.RandomContrast(factor=0.1),
        layers.Lambda(random_saturation),
        layers.Lambda(random_hue),
    ])

def random_saturation(x):
    return tf.image.random_saturation(x, lower=0.5, upper=1.5)

def random_hue(x):
    return tf.image.random_hue(x, max_delta=0.05)

###################################################################################################

augmentation_color = augmentation_layers_color()
augmentation_geometric = augmentation_layers_geometric()

# color-augmentations to the image, geometric-augmentations to image and mask
def augment_image_mask(image, mask, augment_color=True, augment_geometric=True):
    batched_image = tf.expand_dims(image, axis=0)
    batched_mask = tf.expand_dims(mask, axis=0)
    batched_mask = tf.cast(batched_mask, tf.float32)
    
    if augment_color:
        batched_image = augmentation_color(batched_image, training=True)

    if augment_geometric:
        combined = tf.concat([batched_image, batched_mask], axis=-1)
        combined = augmentation_geometric(combined, training=True)
        batched_image, batched_mask = tf.split(combined, [3, tf.shape(batched_mask)[-1]], axis=-1)
    
    batched_image = tf.squeeze(batched_image, axis=0)
    batched_mask = tf.squeeze(batched_mask, axis=0)
    batched_mask = tf.cast(batched_mask, tf.uint8)
    
    return batched_image, batched_mask

# Batch data
def batch(data, augment='none', batch_size=16):
    if augment == 'none':
        return data.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    if augment == 'both':
        cached_train = data.cache()
        return cached_train.map(
            lambda image, mask: augment_image_mask(image, mask, augment_color=True, augment_geometric=True), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    if augment == 'color':
        cached_train = data.cache()
        return cached_train.map(
            lambda image, mask: augment_image_mask(image, mask, augment_color=True, augment_geometric=False), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    if augment == 'geometric':
        cached_train = data.cache()
        return cached_train.map(
            lambda image, mask: augment_image_mask(image, mask, augment_color=False, augment_geometric=True), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)