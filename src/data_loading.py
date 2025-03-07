import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from keras import callbacks
from point_segmentation import random_point, create_gaussian_heatmap
 
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

    def mask_prep(self):
        # Preprocess mask
        self.train_raw = self.train_raw.map(lambda example: mask_preprocessing(example))
        self.val_raw = self.val_raw.map(lambda example: mask_preprocessing(example))
        self.test_raw = self.test_raw.map(lambda example: mask_preprocessing(example))
 
    def heatmaps(self):
        # Generate heatmaps
        self.train_raw = self.train_raw.map(lambda example: heatmap_generation(example))
        self.val_raw = self.val_raw.map(lambda example: heatmap_generation(example))
        self.test_raw = self.test_raw.map(lambda example: heatmap_generation(example))

#############One-hot encoding#############
# Apply one-hot encoding to a single example
def one_hot_encoding(example, num_classes):
    return {
        "image": example["image"],
        "label": tf.one_hot(example["label"], num_classes),
        "segmentation_mask": example["segmentation_mask"],
        "species" : example["species"]
    }
 

#############Resize and normalize#############
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
    seg_mask = example["segmentation_mask"]
    return {
        "image": resize_norm_image(example["image"], resize_shape),
        "label": example['label'],
        "segmentation_mask": resize_mask(seg_mask, resize_shape),
        "species" : example["species"]
    }


#############Mask weird preprocessing#############
# masks: border -> background
def border_to_background(mask):
    threes_mask = tf.equal(mask, 3)
    twos = tf.ones_like(mask) * 2
    mask = tf.where(threes_mask, twos, mask)
    
    return mask

# masks: cat (1) -> 0
def animal_change(mask, species=1):
    if species == 0:
        ones_mask = tf.equal(mask, 1)
        zeros = tf.zeros_like(mask)
        mask = tf.where(ones_mask, zeros, mask)
    
    return mask

def mask_preprocessing(example):
    mask = example["segmentation_mask"]
    species = example["species"]
    label = example['label']
    image = example['image']

    mask = border_to_background(mask)
    mask = animal_change(mask, species)

    return {
        "image": image,
        "label": label,
        "segmentation_mask": mask,
        "species" :species
    }
############Heatmap generation############
def heatmap_generation(example):
    image = example["image"]
    label = example["label"]
    mask = example["segmentation_mask"]
    species = example["species"]

    # get heatmap for random points of animal
    animal_mask = tf.logical_or(tf.equal(mask, 0), tf.equal(mask, 1))
    animal_mask_pos = tf.where(animal_mask == True)
    loc = random_point(animal_mask_pos)
    heatmap = create_gaussian_heatmap(mask_size=mask.shape, point=loc, sigma=20)

    return {
        "image": image,
        "label": label,
        "segmentation_mask": mask,
        #"heatmap": heatmap,
        #"species" : species,
        "image+heatmap": tf.concat([image, heatmap], axis=-1)
    }
###########################################
 
 
def show_examples(train_raw, ds_info):
    """Displays example images from the dataset."""
    tfds.show_examples(train_raw, ds_info, image_key='image')
 
def get_value_counts(ds):
    """Prints the count of each class label in the given dataset."""
    label_list = [label.numpy() for _, label in ds]
    label_counts = pd.Series(label_list).value_counts(sort=True)
    print(label_counts)