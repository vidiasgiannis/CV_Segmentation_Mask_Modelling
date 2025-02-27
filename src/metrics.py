import tensorflow as tf
from tensorflow.keras import backend as K
 
# Intersection over Union (IoU)
def iou_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)  # Convert softmax output to class labels
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64)  # Convert to int64
 
    intersection = tf.reduce_sum(tf.cast(y_pred == y_true, tf.float32))
    union = tf.reduce_sum(tf.cast((y_pred > 0) | (y_true > 0), tf.float32))
 
    return intersection / (union + K.epsilon())  # Avoid division by zero
 
# Dice Coefficient (F1 Score)
def dice_coefficient(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)  # Convert softmax to class labels
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64)  # Ensure y_true is int64
 
    y_true_f = K.flatten(tf.one_hot(y_true, depth=3))
    y_pred_f = K.flatten(tf.one_hot(y_pred, depth=3))
 
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())