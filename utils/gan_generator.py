# utils/gan_generator.py
import tensorflow as tf


def create_generator_model():
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,), kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Reshape((7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(128, 5, strides=(1, 1), padding='same', use_bias=False, kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2DTranspose(64, 5, strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2DTranspose(1, 5, strides=(2, 2), padding='same', use_bias=False, activation='tanh', kernel_initializer=initializer),
    ])
    return model
