import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow as tf

class MeanIoUWrapper(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)
        self.num_classes = num_classes
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probability outputs to integer predictions.
        y_pred = tf.argmax(y_pred, axis=-1)
        # Squeeze y_true if it has an extra channel dimension.
        if len(tf.shape(y_true)) == len(tf.shape(y_pred)) + 1:
            y_true = tf.squeeze(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config
 
# Dice Coefficient (F1 Score)
def dice_coefficient(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)  # Convert softmax to class labels
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64)  # Ensure y_true is int64
 
    y_true_f = K.flatten(tf.one_hot(y_true, depth=3))
    y_pred_f = K.flatten(tf.one_hot(y_pred, depth=3))
 
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())