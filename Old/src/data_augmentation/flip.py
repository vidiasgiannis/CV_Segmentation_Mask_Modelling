import tensorflow as tf

def flip_image(image, mask, horizontal=True, vertical=False):
    """
    Flip the image and mask either horizontally or vertically.

    Parameters:
    image (tensor): The input image to be flipped.
    mask (tensor): The corresponding mask to be flipped.
    horizontal (bool): If True, flip the image and mask horizontally.
    vertical (bool): If True, flip the image and mask vertically.

    Returns:
    tensor: The flipped image and mask.
    """
    # Flip horizontally
    if horizontal:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Flip vertically
    if vertical:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    return image, mask
