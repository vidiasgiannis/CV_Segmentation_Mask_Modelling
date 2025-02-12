import tensorflow as tf

def add_gaussian_noise(image):
    """
    Adds Gaussian noise to a single image tensor.
    
    Parameters:
    - image: A TensorFlow tensor of the image with pixel values scaled between 0 and 1.

    Returns:
    - A TensorFlow tensor of the noisy image with pixel values clipped between 0 and 1.
    """
    # Calculate Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=50/255, dtype=tf.float32)
    noise_img = image + noise
    noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)

    return noise_img