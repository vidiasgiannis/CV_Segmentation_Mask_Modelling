import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from keras import callbacks

# Function to get all augmentation layers together
def augmentation_layers_geometric():
    return keras.Sequential([
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
        layers.RandomBrightness(factor=0.2, value_range=(0,1)),
        layers.RandomContrast(factor=0.1),
        layers.Lambda(lambda x, training=None: random_saturation(x)),
        layers.Lambda(lambda x, training=None: random_hue(x)),
    ])


def augmentation_layers_noise_filter():
    return keras.Sequential([
        layers.GaussianNoise(0.1),
        layers.Lambda(lambda x, training=None: random_blur(x)),
        layers.Lambda(lambda x, training=None: random_jpeg_compression(x)),
    ])



def gaussian_kernel(kernel_size: int, sigma: float):
    """Creates a 2D Gaussian kernel for blurring."""
    ax = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.expand_dims(kernel, axis=-1)  # shape (kernel_size, kernel_size, 1)
    kernel = tf.expand_dims(kernel, axis=-1)  # shape (kernel_size, kernel_size, 1, 1)
    return kernel

def gaussian_blur(x, kernel_size=3, sigma=1.0):
    """Applies Gaussian blur via depthwise convolution."""
    kernel = gaussian_kernel(kernel_size, sigma)
    channels = tf.shape(x)[-1]
    # Tile kernel for all channels
    kernel = tf.tile(kernel, multiples=[1, 1, channels, 1])
    return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')

def motion_blur(x, kernel_size=3):
    """Applies motion blur by convolving with an averaging kernel in a random direction."""
    def horizontal():
        kernel = tf.ones((1, kernel_size), dtype=tf.float32) / kernel_size
        kernel = tf.reshape(kernel, [1, kernel_size, 1, 1])
        channels = tf.shape(x)[-1]
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    def vertical():
        kernel = tf.ones((kernel_size, 1), dtype=tf.float32) / kernel_size
        kernel = tf.reshape(kernel, [kernel_size, 1, 1, 1])
        channels = tf.shape(x)[-1]
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    # Randomly select horizontal or vertical blur
    return tf.cond(tf.random.uniform([], 0, 1) < 0.5, horizontal, vertical)

def random_blur(x):
    """Randomly applies either a Gaussian blur or a motion blur."""
    def apply_gaussian():
        return gaussian_blur(x, kernel_size=3, sigma=1.0)
    def apply_motion():
        return motion_blur(x, kernel_size=3)
    return tf.cond(tf.random.uniform([], 0, 1) < 0.5, apply_gaussian, apply_motion)

def jpeg_compression_single_py(image, quality):
    # Convert quality to a Python int.
    quality = int(quality)
    image_uint8 = tf.image.convert_image_dtype(image, tf.uint8)
    jpeg = tf.io.encode_jpeg(image_uint8, quality=quality)
    image_decoded = tf.io.decode_jpeg(jpeg, channels=3)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image_float

def jpeg_compression_single(image, quality):
    # Wrap the JPEG compression function using tf.py_function.
    result = tf.py_function(func=jpeg_compression_single_py, inp=[image, quality], Tout=tf.float32)
    # Optionally set the shape if known, e.g.:
    result.set_shape(image.shape)
    return result

def random_jpeg_compression(x):
    quality = tf.random.uniform([], minval=30, maxval=70, dtype=tf.int32)
    return tf.map_fn(lambda img: jpeg_compression_single(img, quality), x)



def random_saturation(image):
    return tf.image.random_saturation(image, lower=0.5, upper=1.5)

def random_hue(image):
    return tf.image.random_hue(image, max_delta=0.05)

###################################################################################################

augmentation_color = augmentation_layers_color()
augmentation_geometric = augmentation_layers_geometric()
augmentation_noise_filter = augmentation_layers_noise_filter()

# color-augmentations to the image, geometric-augmentations to image and mask
def augment_image_mask(image, mask, augment_color=None, augment_geometric=None, augment_noise_filter=None):
    batched_image = tf.expand_dims(image, axis=0)
    batched_mask = tf.expand_dims(mask, axis=0)
    batched_mask = tf.cast(batched_mask, tf.float32)
    
    if augment_color:
        batched_image = augmentation_color(batched_image, training=True)

    if augment_geometric:
        combined = tf.concat([batched_image, batched_mask], axis=-1)
        combined = augmentation_geometric(combined, training=True)
        batched_image, batched_mask = tf.split(combined, [3, tf.shape(batched_mask)[-1]], axis=-1)

    if augment_noise_filter:
        batched_image = augmentation_noise_filter(batched_image, training=True)
    
    batched_image = tf.squeeze(batched_image, axis=0)
    batched_mask = tf.squeeze(batched_mask, axis=0)
    batched_mask = tf.cast(batched_mask, tf.uint8)
    
    return batched_image, batched_mask

# Batch data
def batch(data, augment='none', batch_size=16):
    if augment == 'none':
        return data.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
    
    if augment == 'color':
        cached_train = data.cache()
        return cached_train.map(
            lambda image, mask: augment_image_mask(image, mask, augment_color=True, augment_geometric=False, augment_noise_filter=False), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    if augment == 'geometric':
        cached_train = data.cache()
        return cached_train.map(
            lambda image, mask: augment_image_mask(image, mask, augment_color=False, augment_geometric=True, augment_noise_filter=False), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    if augment == 'color+geometric':
        cached_train = data.cache()
        return cached_train.map(
            lambda image, mask: augment_image_mask(image, mask, augment_color=True, augment_geometric=True, augment_noise_filter=False), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    if augment == 'color+geometric+noise_filter':
        cached_train = data.cache()
        return cached_train.map(
            lambda image, mask: augment_image_mask(image, mask, augment_color=True, augment_geometric=True, augment_noise_filter=True), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)