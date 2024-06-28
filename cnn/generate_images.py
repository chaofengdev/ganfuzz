# generate_images.py
import numpy as np
import tensorflow as tf
from utils.gan_generator import create_generator_model
from vector_queue.vector_pool import get_noise

# def get_noise(batch_size=128, noise_dim=100):
#     return np.random.randn(batch_size, noise_dim)


# 用于加载生成器模型并生成图片。
if __name__ == '__main__':
    generator = create_generator_model()
    checkpoint_dir = './gan/logs/dcgan/model/'
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(latest_checkpoint).expect_partial()

    random_vector_for_generation = get_noise()
    generated_images = generator(random_vector_for_generation, training=False)
    generated_images = (generated_images + 1) / 2.0

    np.save('generated_images.npy', generated_images)
