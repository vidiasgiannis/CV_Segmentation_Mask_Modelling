import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


###################
#     task 2a     #
###################
def UNet_model(input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    #Output
    outputs = layers.Conv2D(3, 1, activation='softmax')(conv9)
    model = models.Model(inputs, outputs)
    return model

###################
#     task 2b     #
###################

import tensorflow as tf
from tensorflow.keras import layers, models

class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super(Autoencoder, self).__init__()
        self.encoder = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(1024, (3, 3), activation='relu', padding='same')  # 1024 features
        ])

        self.decoder = models.Sequential([
            layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def build_segmentation_decoder(encoder, num_classes=3):
    inputs = layers.Input(shape=(256, 256, 3))  # Input image

    # Use the encoder to extract features (1024 features in the final layer)
    x = encoder(inputs, training=False)  # Shape: (batch_size, H/16, W/16, 1024)

    # Decoder for segmentation
    x = layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    # Output layer for segmentation
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)

    # Define the model
    model = models.Model(inputs, outputs)
    return model


###################
#     task 2c     #
###################

from tensorflow.keras.layers import Input, Resizing, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, Concatenate, UpSampling2D
from transformers import TFCLIPVisionModel
from tensorflow.keras.models import Model
 
class CLIPEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clip_encoder = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_encoder.trainable = False
 
    def call(self, inputs):
        # The error occurs because the CLIP model expects pixel_values in NCHW format,
        # but TensorFlow typically uses NHWC format
        # Convert from NHWC (batch, height, width, channels) to NCHW (batch, channels, height, width)
        pixel_values = tf.transpose(inputs, [0, 3, 1, 2])
       
        # Now pass the correctly formatted tensor to the CLIP encoder
        outputs = self.clip_encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state
       
    # Add compute_output_shape method to help TensorFlow infer the output shape
    def compute_output_shape(self, input_shape):
        # The CLIP vision model output shape depends on the model configuration
        # For clip-vit-base-patch32, with 224x224 input, the output shape is (batch_size, 50, 768)
        # Where 50 = 49 patches (7x7) + 1 cls token, and 768 is the embedding dimension
        batch_size = input_shape[0]
        return (batch_size, 50, 768)
 
def clip_segmentation_model(input_shape=(128, 128, 3)):
    # Input layer
    input_img = Input(shape=input_shape, name="input_image")
 
    # Preprocessing
    x = Resizing(224, 224, name="resize_input")(input_img)
    def clip_normalize(x):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        # Normalize each channel
        x_normalized = tf.stack([
            (x[:, :, :, 0] - mean[0]) / std[0],
            (x[:, :, :, 1] - mean[1]) / std[1],
            (x[:, :, :, 2] - mean[2]) / std[2]
        ], axis=3)
        return x_normalized
    x = Lambda(clip_normalize, name="clip_normalization")(x)
 
    # Use the custom CLIP encoder layer
    clip_encoder = CLIPEncoderLayer()
    clip_features = clip_encoder(x)
   
    # Reshape the output for decoder
    # Note: The shape will be (batch_size, 50, 768) where 50 = 49 patches (7x7) + 1 cls token
    # We need to adjust the reshape operation to handle this correctly
   
    # Option 1: Skip the cls token and reshape just the patch embeddings
    patch_features = Lambda(lambda x: x[:, 1:, :], name="remove_cls_token")(clip_features)
    features = Reshape((7, 7, 768), name="reshape_features")(patch_features)
   
    # Decoder (you need to implement this part based on your needs)
    # Here's a simple example decoder
    x = Conv2D(256, 3, padding='same', activation='relu')(features)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
   
    # Final output layer (adjust the number of classes as needed)
    outputs = Conv2D(3, 1, activation='softmax')(x)
   
    # Resize output to match the input size if needed
    if input_shape[0] != 224 or input_shape[1] != 224:
        outputs = Resizing(input_shape[0], input_shape[1])(outputs)
 
    return Model(inputs=input_img, outputs=outputs, name="CLIP_Segmentation")


###################
#     task 2d     #
###################

from tensorflow.keras.layers import Input, Resizing, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, Concatenate, UpSampling2D
from transformers import TFCLIPVisionModel
from tensorflow.keras.models import Model
 
class CLIPEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clip_encoder = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_encoder.trainable = False
 
    def call(self, inputs):
        # The error occurs because the CLIP model expects pixel_values in NCHW format,
        # but TensorFlow typically uses NHWC format
        # Convert from NHWC (batch, height, width, channels) to NCHW (batch, channels, height, width)
        pixel_values = tf.transpose(inputs, [0, 3, 1, 2])
       
        # Now pass the correctly formatted tensor to the CLIP encoder
        outputs = self.clip_encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state
       
    # Add compute_output_shape method to help TensorFlow infer the output shape
    def compute_output_shape(self, input_shape):
        # The CLIP vision model output shape depends on the model configuration
        # For clip-vit-base-patch32, with 224x224 input, the output shape is (batch_size, 50, 768)
        # Where 50 = 49 patches (7x7) + 1 cls token, and 768 is the embedding dimension
        batch_size = input_shape[0]
        return (batch_size, 50, 768)
 
def clip_segmentation_model(input_shape=(128, 128, 4)):
    # Input layer for concatenated image and heatmap
    input_total = Input(shape=input_shape, name="input_total")

    # Split into image (first 3 channels) and heatmap (4th channel)
    input_img = Lambda(lambda x: x[..., :3], name="split_image")(input_total)
    input_heatmap = Lambda(lambda x: x[..., 3:], name="split_heatmap")(input_total)

    # Process image through CLIP encoder
    x = Resizing(224, 224, name="resize_image")(input_img)
    
    def clip_normalize(x):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        x_normalized = tf.stack([
            (x[..., 0] - mean[0]) / std[0],
            (x[..., 1] - mean[1]) / std[1],
            (x[..., 2] - mean[2]) / std[2]
        ], axis=-1)
        return x_normalized
    x = Lambda(clip_normalize, name="clip_normalization")(x)
    
    # CLIP encoder
    clip_encoder = CLIPEncoderLayer()
    clip_features = clip_encoder(x)
    
    # Extract patch features and reshape
    patch_features = Lambda(lambda x: x[:, 1:, :], name="remove_cls_token")(clip_features)
    features = Reshape((7, 7, 768), name="reshape_features")(patch_features)

    # Process heatmap: Resize to encoder's spatial dimensions and project channels
    heatmap = Resizing(7, 7, name="resize_heatmap")(input_heatmap)
    heatmap_projected = Conv2D(768, (1, 1), activation='relu', name="heatmap_projection")(heatmap)

    # Integrate heatmap with encoder features
    features_combined = layers.Add(name="features_heatmap_merge")([features, heatmap_projected])

    # Decoder (unchanged from original except input now includes heatmap)
    x = Conv2D(256, 3, padding='same', activation='relu')(features_combined)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)

    # Final output layer
    outputs = Conv2D(1, 1, activation='softmax')(x)

    # Resize output to original input size if needed
    if input_shape[0] != 224 or input_shape[1] != 224:
        outputs = Resizing(input_shape[0], input_shape[1])(outputs)

    return Model(inputs=input_total, outputs=outputs, name="CLIP_Segmentation_with_Heatmap")




