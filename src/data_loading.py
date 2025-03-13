import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from point_segmentation import random_loc, create_gaussian_heatmap, mask_heat_modification
 
class OxfordPetDataset:
    def __init__(self):
        self.train_raw = None
        self.val_raw = None
        self.test_raw = None
        self.test_perturbed = None
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

    def first_perturb(self, std=0):
        self.test_perturbed = self.test_raw.map(lambda example: first_perturbation(example, std))

    def second_perturb(self, n=0):
        self.test_perturbed = self.test_raw.map(lambda example: second_perturbation(example, n))

    def third_perturb(self, a=1.0):
        self.test_perturbed = self.test_raw.map(lambda example: third_perturbation(example, a))

    def fourth_perturb(self, b=0.0):
        self.test_perturbed = self.test_raw.map(lambda example: fourth_perturbation(example, b))

    def fifth_perturb(self, p=0.0):
        self.test_perturbed = self.test_raw.map(lambda example: fifth_perturbation(example, p))

    def sixth_perturb(self, d=0):
        self.test_perturbed = self.test_raw.map(lambda example: sixth_perturbation(example, d))

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
    mask = tf.where(threes_mask, tf.cast(2, mask.dtype), mask)  # Directly replacing 3s with 2s
    
    return mask

# masks: cat (1) -> 0
def animal_change(mask, species=1):
    if species == 0:
        ones_mask = tf.equal(mask, 1)
        mask = tf.where(ones_mask, tf.cast(0, mask.dtype), mask)
    
    return mask

def swap_0_2(mask):
    twos_mask = tf.equal(mask, 2)
    zeros_mask = tf.equal(mask, 0)

    mask = tf.where(twos_mask, tf.cast(0, mask.dtype), mask)
    mask = tf.where(zeros_mask, tf.cast(2, mask.dtype), mask)
    
    return mask


def mask_preprocessing(example):
    mask = example["segmentation_mask"]
    species = example["species"]
    label = example['label']
    image = example['image']

    mask = border_to_background(mask)
    mask = animal_change(mask, species)
    mask = swap_0_2(mask)

    return {
        "image": image,
        "label": label,
        "segmentation_mask": mask,
        "species" :species
    }
############Heatmap generation ---> Mask heat modification############
def heatmap_generation(example):
    image = example["image"]
    label = example["label"]
    original_mask = example["segmentation_mask"]
    species = example["species"]

    # get heatmap for random points of animal
    location = random_loc(mask_shape=original_mask.shape)
    heatmap = create_gaussian_heatmap(mask_shape=original_mask.shape, loc=location, sigma=20)
    mask = mask_heat_modification(original_mask, location)

    return {
        "image": image,
        "label": label,
        "segmentation_mask": mask,
        "original_mask": original_mask,
        "location": location,
        "heatmap": heatmap,
        "species" : species,
        "image+heatmap": tf.concat([image, heatmap], axis=-1)
    }

########Perturbations########
def first_perturbation(example, std=0):
    '''Gaussian pixel noise'''
    image = example["image"]
    shape = tf.shape(image)
    std = std/255 # normalize to [0, 1]
    noise = tf.random.normal(shape=shape, mean=0.0, stddev=std)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return {
        "image": image,
        "segmentation_mask": example["segmentation_mask"]
    }

def second_perturbation(example, n=1):
    '''Gaussian blurring'''
    kernel = tf.constant([
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]
    ], dtype=tf.float32) / 9.0
    # Reshape kernel for 2D convolution [height, width, input_channels, output_channels]
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    image = example["image"]
    if len(image.shape) == 3:  # [height, width, channels]
        image = tf.expand_dims(image, axis=0)  # Add batch dimension [1, height, width, channels]
        
        for _ in range(n):
            channels = []
            for i in range(image.shape[-1]):
                channel = image[..., i:i+1]  # Extract single channel and keep dimensions
                blurred = tf.nn.conv2d(channel, filters=kernel, strides=[1, 1, 1, 1], padding="SAME")
                channels.append(blurred)
            image = tf.concat(channels, axis=-1)
        
        image = tf.squeeze(image, axis=0)  # Remove batch dimension
    else:
        image = tf.expand_dims(tf.expand_dims(image, axis=0), axis=-1)  # [1, height, width, 1]
        
        for _ in range(n):
            image = tf.nn.conv2d(image, filters=kernel, strides=[1, 1, 1, 1], padding="SAME")
        image = tf.squeeze(tf.squeeze(image, axis=0), axis=-1)  # Remove added dimensions
    image = tf.clip_by_value(image, 0.0, 1.0)  # Ensure valid pixel range

    return {
        "image": image,
        "segmentation_mask": example["segmentation_mask"]
    }

def third_perturbation(example, a=1.0):
    '''Contrast increase/decrease'''
    image = example["image"]
    image = tf.clip_by_value(image * a, 0.0, 1.0)  # Ensure valid pixel range
    
    return {
        "image": image,
        "segmentation_mask": example["segmentation_mask"]
    }

def fourth_perturbation(example, b=0.0):
    '''Brightness increase/decrease'''
    b = b / 255.0  # Normalize to [0, 1]
    image = example["image"]
    image = tf.clip_by_value(image + b, 0.0, 1.0)  # Ensure valid pixel range
    
    return {
        "image": image,
        "segmentation_mask": example["segmentation_mask"]
    }

def fifth_perturbation(example, p=0.0):
    '''Occlusion of the Image Increase'''
    image = example["image"]
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    if p == 0:
        pass
    else:
        p = tf.cast(p, tf.int32)
        y = tf.random.uniform(shape=[], minval=0, maxval=tf.maximum(1, height - p), dtype=tf.int32)
        x = tf.random.uniform(shape=[], minval=0, maxval=tf.maximum(1, width - p), dtype=tf.int32)
        p = tf.minimum(p, tf.minimum(height, width))
    
        # Create meshgrid for the occlusion area
        y_indices = tf.range(y, y + p)
        x_indices = tf.range(x, x + p)
        
        # Create grid coordinates for the occlusion
        indices = tf.stack(tf.meshgrid(y_indices, x_indices, indexing='ij'), axis=-1)
        indices = tf.reshape(indices, [-1, 2])
        
        # Add channel dimension to indices
        channel_indices = tf.tile(tf.expand_dims(indices, 1), [1, 3, 1])
        channel_indices = tf.concat([
            channel_indices[:, 0, :], 
            tf.expand_dims(tf.zeros_like(indices[:, 0]), 1)
        ], axis=1)
        channel_indices = tf.concat([
            channel_indices[:, :2], 
            tf.expand_dims(tf.ones_like(indices[:, 0]), 1)
        ], axis=1)
        channel_indices = tf.concat([
            channel_indices[:, :2], 
            tf.expand_dims(tf.ones_like(indices[:, 0]) + tf.ones_like(indices[:, 0]), 1)
        ], axis=1)
        mask = tf.ones_like(image)
        zeros_patch = tf.zeros([p, p, 3])
        indices = tf.stack([
            tf.repeat(tf.range(y, y + p), p),
            tf.tile(tf.range(x, x + p), [p])
        ], axis=1)
        zeros_flat = tf.zeros([p * p, 3])
        mask = tf.tensor_scatter_nd_update(mask, indices, zeros_flat)
        image = image * mask
    
    return {
        "image": image,
        "segmentation_mask": example["segmentation_mask"]
    }

def sixth_perturbation(example, d=0):
    '''Salt and Pepper Noise'''
    image = example["image"]
    
    salt_mask = tf.random.uniform(tf.shape(image)) < (d / 2.0)
    pepper_mask = tf.random.uniform(tf.shape(image)) < (d / 2.0)
    
    noisy_image = tf.where(salt_mask, tf.ones_like(image), image)
    noisy_image = tf.where(pepper_mask, tf.zeros_like(image), noisy_image)
    
    return {
        "image": noisy_image,
        "segmentation_mask": example["segmentation_mask"]
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
