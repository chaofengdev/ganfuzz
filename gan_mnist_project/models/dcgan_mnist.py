#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf  # TF2
from IPython import display

# 请注意，这里gan_trainer必须与模型训练器的文件名称一致
import gan_trainer
# from . import gan_trainer

# 确保使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置 GPU 内存增长  --这里都是为了缓解gpu显存不够的权宜之计。
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

logging.getLogger("tensorflow").setLevel(logging.ERROR)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_generator_model():
    """
    生成模型。

    Returns:
      由序列模型构建的生成模型。
    """
    # 所有参数都采用均值为0、方差为0.02的正态分布随机数来初始化
    initializer = tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02)

    # 采用顺序模型（Sequential，又称为序列模型）来构建生成模型
    model = tf.keras.Sequential([
        # 第一个全连接层，从随机噪声连接到12544（=7×7×256）个神经元
        tf.keras.layers.Dense(
            7*7*256, use_bias=False,
            input_shape=(100,),
            kernel_initializer=initializer),
        # 批量正则化
        tf.keras.layers.BatchNormalization(),
        # 采用alpha=0.2的带泄露的ReLU激活函数
        tf.keras.layers.LeakyReLU(alpha=0.2),

        # 将12544（=7×7×256）个神经元整型为形状为7×7×256的张量
        # 为转置卷积操作做准备
        tf.keras.layers.Reshape((7, 7, 256)),

        # 第二个转置卷积层，输入7×7×256，输出7×7×128
        tf.keras.layers.Conv2DTranspose(
            128, 5, strides=(1, 1), padding='same',
            use_bias=False, kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),

        # 第三个转置卷积层，输入7×7×128，输出14×14×64
        tf.keras.layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding='same',
            use_bias=False, kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),

        # 第四个转置卷积层，输入14×14×64，输出28×28×1
        tf.keras.layers.Conv2DTranspose(
            1, 5, strides=(2, 2), padding='same', use_bias=False,
            activation='tanh', kernel_initializer=initializer),
    ])

    return model


def create_discriminator_model():
    """
    辨别模型.

    Returns:
      采用序列模型构建的辨别模型
    """
    # 初始化器，采用均值为0、方差为0.02的正态分布随机数来初始化
    initializer = tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.02)

    # 采用顺序模型构建辨别模型
    model = tf.keras.Sequential([

        # 第一个卷积层，输入28×28×1，输出14×14×64
        tf.keras.layers.Conv2D(
            64, 5, strides=(2, 2), padding='same',
            kernel_initializer=initializer),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),

        # 第二个卷积层，输入14×14×64，输出7×7×128
        tf.keras.layers.Conv2D(
            128, 5, strides=(2, 2), padding='same',
            kernel_initializer=initializer),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),

        # 第三层，全连接层。展平，为了便于跟之后的全连接层连接
        tf.keras.layers.Flatten(),

        # 第四层，输出层。只有1个神经元，输出1代表输入张量来自真实样本；
        # 输出0代表输入张量来源于生成模型
        tf.keras.layers.Dense(
            1, kernel_initializer=initializer, activation='sigmoid')
    ])

    return model


def read_mnist(buffer_size, batch_size):
    """
     读取MNIST数据集。

    参数：
        buffer_size：乱序排列时，乱序的缓存大小。
        batch_size：批处理的大小
    Return：
        训练样本数据集。
    """
    # 读取MNIST样本数据
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

    # 将MNIST样本数据，按照28×28×1的形状整理
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')

    # 将MNSIT的像素取值区间从[0, 255]映射到[-1， 1]区间，
    train_images = (train_images - 127.5) / 127.5

    # 对样本数据进行乱序排列、并按照batch_size大小划分成不同批次数据
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(buffer_size).batch(batch_size)
    return train_dataset


def save_images(gen_samples, step, image_dir):
    """
    利用生成模型图片，然后，保存到指定的文件夹下。

    参数：
        generator   : 生成模型，已经经过epoch轮训练。
        epoch       : 训练的轮数。
        test_input  : 测试用的随机噪声。
        image_dir   : 用于存放生成图片的路径。
    """

    display.clear_output(wait=True)

    plt.figure(figsize=(4, 4))
    for i in range(gen_samples.shape[0]):
        plt.subplot(4, 4, i+1)

        # 将生成的数据取值范围映射回到MNIST图像的像素取值范围[0, 255]
        plt.imshow(gen_samples[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    # 逐级创建目录，如果目录已存在，则忽略
    os.makedirs(image_dir, exist_ok=True)
    image_file_name = os.path.join(image_dir, 'image_step_{:05d}.png')
    plt.savefig(image_file_name.format(step))
    plt.close('all')  # 关闭图


def main(epochs=10, buffer_size=10000, batch_size=128,
         save_checkpoints_steps=100, save_image_steps=1000):
    """
    模型训练的入口函数。

    参数：
        epochs：训练的轮数，每一轮的训练使用全部样本数据一次。
        buffer_size：乱序排列时，乱序的缓存大小。
        batch_size：批处理的大小
    Return：
        训练样本数据集。
    """

    # 读取训练样本数据集
    train_dataset = read_mnist(buffer_size, batch_size)

    # 构建GAN模型训练器
    trainer = gan_trainer.GANTrainer(
        # 生成模型构建函数
        generator=create_generator_model,
        # 生成模型的优化器
        generator_optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5),
        # 辨别模型构建函数
        discriminator=create_discriminator_model,
        # 辨别模型的优化器
        discriminator_optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5),
        name='dcgan'
    )
    print('\n开始训练 ...')
    return trainer.train(train_dataset, epochs,
                         save_checkpoints_steps=save_checkpoints_steps,
                         save_image_steps=save_image_steps,
                         save_image_func=save_images)


# 入口函数，训练40轮次
if __name__ == '__main__':
    main(30)  # 受限于笔记本和时间成本，这里暂时跑20轮，查看效果。

# 训练轮次：30， buffer_size=10000, 批次大小：256
# main(epochs=30, buffer_size=10000, batch_size=256)
