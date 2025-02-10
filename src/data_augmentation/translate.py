import tensorflow as tf
import numpy as np
from math import ceil, floor

def get_translate_parameters(index, IMAGE_SIZE):
    """
    Get the translation parameters for each direction (left, right, top, bottom).
    """
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start, w_end = 0, ceil(0.8 * IMAGE_SIZE)
        h_start, h_end = 0, IMAGE_SIZE
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start, w_end = floor((1 - 0.8) * IMAGE_SIZE), IMAGE_SIZE
        h_start, h_end = 0, IMAGE_SIZE
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start, w_end = 0, IMAGE_SIZE
        h_start, h_end = 0, ceil(0.8 * IMAGE_SIZE)
    else:  # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start, w_end = 0, IMAGE_SIZE
        h_start, h_end = floor((1 - 0.8) * IMAGE_SIZE), IMAGE_SIZE

    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs, X_masks, IMAGE_SIZE):
    """
    Translate the input image and its corresponding mask in four directions (left, right, top, bottom).
    """
    translated_images = []
    translated_masks = []

    # Loop through all 4 translation directions (left, right, top, bottom)
    for i in range(4):
        # Get translation parameters for the current direction
        offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i, IMAGE_SIZE)

        # Translate the image and mask by cropping and padding them
        translated_image = tf.image.crop_to_bounding_box(X_imgs, h_start, w_start, size[0], size[1])
        translated_mask = tf.image.crop_to_bounding_box(X_masks, h_start, w_start, size[0], size[1])

        # Resize the cropped image and mask back to the original size
        translated_image = tf.image.resize_with_crop_or_pad(translated_image, IMAGE_SIZE, IMAGE_SIZE)
        translated_mask = tf.image.resize_with_crop_or_pad(translated_mask, IMAGE_SIZE, IMAGE_SIZE)

        # Append translated images and masks
        translated_images.append(translated_image)
        translated_masks.append(translated_mask)

    return translated_images, translated_masks