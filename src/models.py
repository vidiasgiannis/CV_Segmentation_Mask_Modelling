import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = models.Model(inputs, outputs)
    return model


###################
#     task 2b     #
###################

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, BatchNormalization, Activation,
    MaxPool2D, Input, Concatenate
)
from tensorflow.keras.models import Model

# ---------------------------- #
# ✅ Convolutional Block
# ---------------------------- #
def conv_block(input, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(input)
    x = BatchNormalization()(x)   
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)  
    x = Activation("relu")(x)

    return x

# ---------------------------- #
# ✅ Encoder Block (Reduced Depth)
# ---------------------------- #
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)  # ✅ Downsampling step
    return x, p    

# ---------------------------- #
# ✅ Decoder Block (Reduced Depth)
# ---------------------------- #
def decoder_block(input, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = conv_block(x, num_filters)
    return x

# ---------------------------- #
# ✅ Encoder (Max Depth: 512)
# ---------------------------- #
def build_encoder(input_img):
    s1, p1 = encoder_block(input_img, 32)  # ✅ Reduce filters
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    encoded = conv_block(p4, 512)  # ✅ Max filters at 512
    return encoded

# ---------------------------- #
# ✅ Decoder (Reconstruct Image)
# ---------------------------- #
def build_decoder(encoded):
    d1 = decoder_block(encoded, 256)
    d2 = decoder_block(d1, 128)
    d3 = decoder_block(d2, 64)
    d4 = decoder_block(d3, 32)

    decoded = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(d4)  # ✅ Final output
    return decoded

# ---------------------------- #
# ✅ Autoencoder (Encoder + Decoder)
# ---------------------------- #
def build_autoencoder(input_shape=(256, 256, 3)):
    input_img = Input(shape=input_shape)
    encoded_features = build_encoder(input_img)  # ✅ Extract spatial features
    decoded_output = build_decoder(encoded_features)  # ✅ Reconstruct image

    autoencoder = Model(input_img, decoded_output, name="Autoencoder")
    return autoencoder

# ---------------------------- #
# ✅ Encoder Model (Standalone)
# ---------------------------- #
def build_encoder_model(input_shape=(256, 256, 3)):
    input_img = Input(shape=input_shape)
    encoded_features = build_encoder(input_img)
    
    encoder = Model(input_img, encoded_features, name="Encoder")
    return encoder

# ---------------------------- #
# ✅ Decoder Block for U-Net (Skip Connections)
# ---------------------------- #
def decoder_block_for_unet(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])  # ✅ Add skip connections
    x = conv_block(x, num_filters)
    return x

# ---------------------------- #
# ✅ U-Net Model (Max Depth: 512)
# ---------------------------- #
def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # ✅ Encoder (Keep Skip Connections)
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    # ✅ Bridge (Max Filters: 512)
    b1 = conv_block(p4, 512)

    # ✅ Decoder (Skip Connections)
    d1 = decoder_block_for_unet(b1, s4, 256)
    d2 = decoder_block_for_unet(d1, s3, 128)
    d3 = decoder_block_for_unet(d2, s2, 64)
    d4 = decoder_block_for_unet(d3, s1, 32)

    # ✅ Final Segmentation Output
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model