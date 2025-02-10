import numpy as np

def random_crop(image, mask, crop_height, crop_width):
    """
    Randomly crops an image and a mask to the specified dimensions.

    Parameters:
    image (np.array): The input image as a NumPy array of shape (H, W, 3).
    mask (np.array): The corresponding mask as a NumPy array of shape (H, W, 1).
    crop_height (int): The desired height of the crop.
    crop_width (int): The desired width of the crop.

    Returns:
    np.array: The cropped image.
    np.array: The cropped mask.
    """
    assert image.shape[0] >= crop_height, "Crop height can't be more than image height"
    assert image.shape[1] >= crop_width, "Crop width can't be more than image width"
    
    # Choose a random start point for the crop
    max_y = image.shape[0] - crop_height
    max_x = image.shape[1] - crop_width
    start_y = np.random.randint(0, max_y)
    start_x = np.random.randint(0, max_x)

    # Crop the image and mask using the same start point
    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    cropped_mask = mask[start_y:start_y + crop_height, start_x:start_x + crop_width]

    return cropped_image, cropped_mask
