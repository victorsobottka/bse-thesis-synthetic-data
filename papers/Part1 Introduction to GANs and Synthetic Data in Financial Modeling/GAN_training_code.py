
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel('ERROR')  # Only show errors

# Generator model
def create_generator():
    input_layer = layers.Input(shape=(100,))
    x = layers.Dense(128, activation="relu")(input_layer)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    output_layer = layers.Dense(784, activation="tanh")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Discriminator model
def create_discriminator():
    input_layer = layers.Input(shape=(784,))
    x = layers.Dense(512, activation="relu")(input_layer)
    x = layers.Dense(256, activation="relu")(x)
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# GAN model to combine generator and discriminator
def create_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze discriminator during GAN training
    gan_input = layers.Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan_model = Model(inputs=gan_input, outputs=gan_output)
    return gan_model

# Function to train the GAN
def train_gan(generator, discriminator, gan, data, epochs=10000, batch_size=64):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Train Discriminator
        noise = np.random.normal(0, 1, (half_batch, 100))
        generated_data = generator.predict(noise, verbose=0)

        real_data = data[np.random.randint(0, data.shape[0], half_batch)]

        # Train discriminator on real and fake data
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")

# Prepare data
data = np.random.normal(0, 1, (1000, 784))

# Initialize models
generator = create_generator()
discriminator = create_discriminator()
discriminator.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
gan = create_gan(generator, discriminator)
gan.compile(optimizer=Adam(), loss="binary_crossentropy")

# Train GAN
train_gan(generator, discriminator, gan, data, epochs=10000, batch_size=64)
