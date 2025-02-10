import tensorflow as tf

def apply_color_jitter(images, brightness=0, contrast=0, saturation=0, hue=0):
    if brightness != 0:
        images = tf.image.random_brightness(images, max_delta=brightness)
    if contrast != 0:
        images = tf.image.random_contrast(images, lower=1-contrast, upper=1+contrast)
    if saturation != 0:
        images = tf.image.random_saturation(images, lower=1-saturation, upper=1+saturation)
    if hue != 0:
        images = tf.image.random_hue(images, max_delta=hue)
    return images
