import tensorflow as tf

def random_point(animal_mask_pos):
    shuffled_indices = tf.random.shuffle(animal_mask_pos)
    loc = shuffled_indices[0]
    
    return loc

def create_gaussian_heatmap(mask_size, point, sigma=10.0):
    """
    Args:
        image_size: Tuple of (height, width) representing the image dimensions
        point: Tensor of shape (2,) containing (x, y) coordinates of the center
        sigma: Standard deviation of the Gaussian kernel (controls the spread)
    
    Returns:
        A tensor of shape (height, width, 1) containing the heatmap
    """
    height, width, channels = mask_size

    y_range = tf.range(height, dtype=tf.float32)
    x_range = tf.range(width, dtype=tf.float32)
    y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing='ij')

    # Swap the order: point[0] is y, point[1] is x.
    y_point = tf.cast(point[0], tf.float32)
    x_point = tf.cast(point[1], tf.float32)

    # Calculate gaussian heatmap
    squared_dist = tf.square(x_grid - x_point) + tf.square(y_grid - y_point)
    heatmap = tf.exp(-squared_dist / (2.0 * sigma * sigma))

    heatmap = tf.reshape(heatmap, (height, width, channels))
    return heatmap