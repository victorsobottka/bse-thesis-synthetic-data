
import tensorflow as tf
from tensorflow.keras import layers

# Define the Generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='tanh') # Output size to match the data shape
    ])
    return model

# Define the Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(1,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Output is a probability
    ])
    return model

# Compile GAN with Generator and Discriminator
generator = build_generator()
discriminator = build_discriminator()

# Combine the models in the adversarial network
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')
