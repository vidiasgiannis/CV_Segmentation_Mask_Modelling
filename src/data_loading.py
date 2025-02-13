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
            as_supervised=True,
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
        self.train = self.train_raw.map(lambda image, label: (image, tf.one_hot(label, self.num_classes)))
        self.val = self.val_raw.map(lambda image, label: (image, tf.one_hot(label, self.num_classes)))
        self.test = self.test_raw.map(lambda image, label: (image, tf.one_hot(label, self.num_classes)))


def show_examples(train_raw, ds_info):
    """Displays example images from the dataset."""
    tfds.show_examples(train_raw, ds_info, image_key='image')

def get_value_counts(ds):
    """Prints the count of each class label in the given dataset."""
    label_list = [label.numpy() for _, label in ds]
    label_counts = pd.Series(label_list).value_counts(sort=True)
    print(label_counts)

def view_image(dataset, ds_info, get_label_name):
    # Build the custom function to display image and label name
    image, label = next(iter(dataset))
    print('Image shape: ', image.shape)
    plt.imshow(image)
    _ = plt.title(get_label_name(label))


# resize and normalize images
def resize_normalize(data, input_shape=(128, 128)):
    # Resize and normalize
    image = tf.image.resize(data['image'], input_shape)
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.image.resize(data['segmentation_mask'], input_shape)
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask
