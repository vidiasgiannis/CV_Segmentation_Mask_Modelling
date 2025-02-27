import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from keras import callbacks
 
class OxfordPetDataset:
    def __init__(self):
        self.train_raw = None
        self.val_raw = None
        self.test_raw = None
        self.ds_info = None
        self.num_classes = None
        self.num_train_examples = 0
        self.num_val_examples = 0
        self.num_test_examples = 0
        self.get_label_name = None
 
    def load_data(self):
        """Loads the Oxford-IIIT Pet dataset."""
        (self.train_raw, self.val_raw, self.test_raw), self.ds_info = tfds.load(
            name='oxford_iiit_pet',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=False, # with False each example is a dictionary ['image', 'label', 'segmentation_mask']
            with_info=True
        )
       
        self.num_classes = self.ds_info.features['label'].num_classes
 
        # Function to obtain the name for the label integer
        self.get_label_name = self.ds_info.features['label'].int2str
       
        print(f'Number of classes: {self.num_classes}')
 
        self.num_train_examples = tf.data.experimental.cardinality(self.train_raw).numpy()
        self.num_val_examples = tf.data.experimental.cardinality(self.val_raw).numpy()
        self.num_test_examples = tf.data.experimental.cardinality(self.test_raw).numpy()
 
        print(f'Number of training samples: {self.num_train_examples}')
        print(f'Number of validation samples: {self.num_val_examples}')
        print(f'Number of test samples: {self.num_test_examples}')
 
    def one_hot_encoding(self):
        # Apply one-hot encoding to all sets
        self.train_raw = self.train_raw.map(lambda example: one_hot_encoding(example, self.num_classes))
        self.val_raw = self.val_raw.map(lambda example: one_hot_encoding(example, self.num_classes))
        self.test_raw = self.test_raw.map(lambda example: one_hot_encoding(example, self.num_classes))
 
    def res_norm(self, resize_shape=(128, 128)):
        # Resize and normalize
        self.train_raw = self.train_raw.map(lambda example: resize_norm(example, resize_shape))
        self.val_raw = self.val_raw.map(lambda example: resize_norm(example, resize_shape))
        self.test_raw = self.test_raw.map(lambda example: resize_norm(example, resize_shape))
 
# one-hot encoding of a dataset
"""Converts the label of data dict to one-hot encoding while preserving the dictionary structure."""
 
def one_hot_encoding(example, num_classes):
    return {
        "image": example["image"],
        "label": tf.one_hot(example["label"], num_classes),
        "segmentation_mask": example["segmentation_mask"]
    }
 
# single image resize by bilinear interpolation
def resize_norm_image(image, resize_shape):
    image = tf.image.resize(image, resize_shape, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) / 255.0
    return image
 
# single mask resize by nearest neighbor
def resize_mask(mask, resize_shape):
    """Resize and normalize a single image."""
    mask = tf.image.resize(mask, resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return mask
 
# resize and normalize a dataset
def resize_norm(example, resize_shape):
    # Resize and normalize
    seg_mask = example["segmentation_mask"] -1 # Subtract 1 to make the classes start from 0
    return {
        "image": resize_norm_image(example["image"], resize_shape),
        "label": example['label'],
        "segmentation_mask": resize_mask(seg_mask, resize_shape)
    }
 
 
def show_examples(train_raw, ds_info):
    """Displays example images from the dataset."""
    tfds.show_examples(train_raw, ds_info, image_key='image')
 
def get_value_counts(ds):
    """Prints the count of each class label in the given dataset."""
    label_list = [label.numpy() for _, label in ds]
    label_counts = pd.Series(label_list).value_counts(sort=True)
    print(label_counts)