import tensorflow as tf
from tensorflow.keras import backend as K

class MeanIoUWrapper(tf.keras.metrics.MeanIoU):
    def _init_(self, num_classes, name='mean_iou', **kwargs):
        super()._init_(num_classes=num_classes, name=name, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probability outputs to integer predictions
        y_pred = tf.argmax(y_pred, axis=-1)

        # ✅ FIX: Dynamically check and adjust shape
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)

        # Ensure y_true has the same rank as y_pred
        if y_true_shape.shape.rank is not None and y_true_shape.shape.rank > y_pred_shape.shape.rank:
            y_true = tf.squeeze(y_true, axis=-1)

        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

def dice_coefficient(y_true, y_pred, num_classes=3):
    """
    Computes the Dice Coefficient for multi-class segmentation.

    Args:
        y_true: Ground truth labels (integer labels).
        y_pred: Predicted probabilities (softmax output).
        num_classes: Number of classes (default is 3 for your case).

    Returns:
        Average Dice coefficient across all classes.
    """
    y_pred = tf.argmax(y_pred, axis=-1)  # Convert softmax probabilities to class labels
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)  # Ensure y_true is int32

    # One-hot encode y_true and y_pred
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
    y_pred_one_hot = tf.one_hot(y_pred, depth=num_classes, dtype=tf.float32)

    # Compute Dice score per class
    intersection = K.sum(y_true_one_hot * y_pred_one_hot, axis=[0, 1, 2])  # Sum over spatial dimensions
    union = K.sum(y_true_one_hot, axis=[0, 1, 2]) + K.sum(y_pred_one_hot, axis=[0, 1, 2])

    dice_per_class = (2.0 * intersection) / (union + K.epsilon())

    # Compute mean Dice score across all classes
    return K.mean(dice_per_class)