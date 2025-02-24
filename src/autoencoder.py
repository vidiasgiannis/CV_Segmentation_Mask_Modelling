import tensorflow as tf
from tensorflow.keras import layers, models

# Autoencoder Model
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
            layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        ])

        self.decoder = models.Sequential([
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

    # Define a segmentation decoder
def build_segmentation_decoder(encoder):
    inputs = layers.Input(shape=(256, 256, 3))  # Input image size

    # Use the pretrained encoder
    x = encoder(inputs, training=False)  

    # Decoder (Upsampling layers to reconstruct segmentation mask)
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Output segmentation mask (1 channel, sigmoid for binary segmentation)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    # Create final model
    model = models.Model(inputs, outputs)
    return model


