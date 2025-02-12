import tensorflow as tf

def rotate_image(image, mask):
    """
    Rotate the image and mask by 0, 90, 180, and 270 degrees.

    Parameters:
    image (tensor): The input image to be rotated.
    mask (tensor): The corresponding mask to be rotated.

    Returns:
    tuple: Four rotated images and masks.
    """
    # Create a list to store the rotated images and masks
    rotated_images = []
    rotated_masks = []
    
    # Rotate by 0, 90, 180, and 270 degrees
    for k in range(4):
        rotated_image = tf.image.rot90(image, k=k)  # Rotate the image
        rotated_mask = tf.image.rot90(mask, k=k)    # Rotate the mask
        rotated_images.append(rotated_image)
        rotated_masks.append(rotated_mask)
    
    return rotated_images, rotated_masks
