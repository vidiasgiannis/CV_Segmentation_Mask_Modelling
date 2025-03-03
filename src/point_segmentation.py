import tensorflow as tf

def random_point(image_size):
    height, width = image_size
    
    x = tf.random.uniform(shape=(), maxval=width, dtype=tf.int32)
    y = tf.random.uniform(shape=(), maxval=height, dtype=tf.int32)
    
    point = tf.stack([x, y])
    
    return point

def create_gaussian_heatmap(image_size, point, sigma=10.0):
    """
    Args:
        image_size: Tuple of (height, width) representing the image dimensions
        point: Tensor of shape (2,) containing (x, y) coordinates of the center
        sigma: Standard deviation of the Gaussian kernel (controls the spread)
    
    Returns:
        A tensor of shape (height, width, 1) containing the heatmap
    """
    height, width = image_size
    
    y_range = tf.range(height, dtype=tf.float32)
    x_range = tf.range(width, dtype=tf.float32)
    y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing='ij')
    x_point = tf.cast(point[0], tf.float32)
    y_point = tf.cast(point[1], tf.float32)
    
    # Calculate gaussian heatmap
    squared_dist = tf.square(x_grid - x_point) + tf.square(y_grid - y_point)
    heatmap = tf.exp(-squared_dist / (2.0 * sigma * sigma))
    
    # Reshape to (height, width, 1) for compatibility with image processing functions
    heatmap = tf.reshape(heatmap, (height, width, 1))
    
    return heatmap