import tensorflow as tf

def random_loc(mask_shape):
    height, width, _ = mask_shape

    y = tf.random.uniform((), 0, height, dtype=tf.int32)
    x = tf.random.uniform((), 0, width, dtype=tf.int32)

    return tf.stack([y, x])
    

def create_gaussian_heatmap(mask_shape, loc, sigma=10.0):
    """
    Args:
        image_size: Tuple of (height, width) representing the image dimensions
        point: Tensor of shape (2,) containing (x, y) coordinates of the center
        sigma: Standard deviation of the Gaussian kernel (controls the spread)

    Returns:
        A tensor of shape (height, width, 1) containing the heatmap
    """
    height, width, channels = mask_shape

    y_range = tf.range(height, dtype=tf.float32)
    x_range = tf.range(width, dtype=tf.float32)
    y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing='ij')

    # Swap the order: point[0] is y, point[1] is x.
    y_point = tf.cast(loc[0], tf.float32)
    x_point = tf.cast(loc[1], tf.float32)

    # Calculate gaussian heatmap
    squared_dist = tf.square(x_grid - x_point) + tf.square(y_grid - y_point)
    heatmap = tf.exp(-squared_dist / (2.0 * sigma * sigma))

    heatmap = tf.reshape(heatmap, (height, width, channels))
    return heatmap

def mask_heat_modification(mask, loc):
    loc_label = mask[loc[0], loc[1], 0]
    region_interest_mask = tf.equal(mask, loc_label)
    return tf.cast(region_interest_mask, tf.int32)