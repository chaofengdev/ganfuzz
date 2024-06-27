import os
import matplotlib.pyplot as plt
import tensorflow as tf
from vector_queue.vector_pool import get_noise


# 显示生成的图片并保存
def display_images(images, num_examples, save_dir=None):
    plt.figure(figsize=(6, 6))  # 调整整体画布大小
    rows = int(num_examples ** 0.5)  # 计算行数
    cols = int(num_examples ** 0.5)  # 计算列数
    total_plots = rows * cols
    for i in range(min(num_examples, total_plots)):
        plt.subplot(rows, cols, i + 1)  # 使用计算后的行数和列数
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.tight_layout()  # 自动调整子图间距

    # 如果指定了保存目录，则保存图片
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'generated_images.png')
        plt.savefig(save_path)
        print(f'保存生成的图片到: {save_path}')
    else:
        plt.show()


# 定义生成器模型
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


# 要使用已经训练好的GAN模型进行推理，需要加载保存的模型权重，并使用生成器模型生成图片。
if __name__ == '__main__':

    # 加载最新的检查点
    checkpoint_dir = './gan/logs/dcgan/model/'
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    # 创建生成器并恢复检查点
    generator = create_generator_model()
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(latest_checkpoint).expect_partial()

    # 随机生成输入噪声
    # noise_dim = 100
    # num_examples = 16
    # random_vector_for_generation = tf.random.normal([num_examples, noise_dim])
    # 重构：从种子队列中，按照一定的策略选择种子，经过变异算法后形成新的种子，多个种子组织成为张量
    random_vector_for_generation = get_noise()

    # 生成图片
    generated_images = generator(random_vector_for_generation, training=False)
    # 显示图片
    display_images(generated_images, 128, save_dir='save')
