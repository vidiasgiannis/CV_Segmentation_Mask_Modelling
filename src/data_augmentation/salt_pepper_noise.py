
import tensorflow as tf
def salt_and_pepper(image, prob_salt=0.01, prob_pepper=0.01):
    # Create a random matrix with the same shape as the image
    rnd = tf.random.uniform(shape=tf.shape(image))
    
    # Salt noise (setting pixels to 1)
    salt_mask = tf.cast(rnd < prob_salt, tf.float32)
    noisy_image = tf.where(salt_mask == 1, 1, image)
    
    # Pepper noise (setting pixels to 0)
    pepper_mask = tf.cast(rnd > 1 - prob_pepper, tf.float32)
    noisy_image = tf.where(pepper_mask == 1, 0, noisy_image)
    
    return noisy_image