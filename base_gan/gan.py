#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import imageio
import glob
import IPython

from IPython import display
import tensorflow as tf  # TensorFlow 2.0

# 导入 全连接层、批量标准化层、带泄露的激活函数、激活函数
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Activation

# 确保TensorFlow 2.0环境
assert tf.__version__.startswith('2')


def build_generator_model():
    """
    构建能生成MNIST图片的生成模型（G），采用全连接神经网络
    输入张量是代表随机噪声的1维向量，长度为100；
    输出是长度28×28×1的张量，与MNIST中图像尺寸一致。

    模型各层：输入层->隐藏层(h1)->隐藏层（h2）->输出层。

    返回值:
      生成模型（G）
    """
    # 顺序模型，从输入层到输出层逐层堆叠神经网络层
    generator = tf.keras.Sequential()

    # 第一个隐藏层，神经元数量：128 个
    # input_shape代表输入的随机噪声，形状是1维张量，长度为100
    generator.add(Dense(128, input_shape=(100,),  name="h0"))
    generator.add(LeakyReLU(0.2))

    # 第二个隐藏层，神经元数量256个，线性转换后，在执行批量标准化和激活
    generator.add(Dense(256, name="h1"))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # 第三个隐藏层，神经元数量512个
    generator.add(Dense(512, name="h2"))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # 第四个隐藏层，神经元数量1024个
    generator.add(Dense(1024, name="h3"))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    # 输出层，总共有784个神经元，每个神经元对应一个像素（784=28×28×1）
    generator.add(Dense(784, name="output"))

    # 请注意：这里的激活函数必须是'tanh'，把各个元素取值映射到[-1, 1]区间
    # 然后，再将 原始值 × 127.5 + 127.5，将数据映射到[0, 255]，
    # [0, 255] MNIST图像像素的取值区间，这样才能最终正确的生成图片
    generator.add(Activation('tanh'))
    return generator


def build_discriminator_model():
    """
    构建辨别模型(D)，输入张量是MNIST数据的图片，形状为28×28×1。
    输出的是输入图像所属的类别（来源于真实样本、还是由生成模型生成的样本）

    在这里，我们展示创建模型的另外一种编程语法。

    返回值:
      辨别模型（D）
    """
    discriminator = tf.keras.models.Sequential([
        # 输入层：28×28，将输入张量展平，便于与后面的各层连接
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        # 第一个隐藏层,有512个神经元，采用LeakyReLU激活函数激活
        tf.keras.layers.Dense(512,   name='h0'),
        tf.keras.layers.LeakyReLU(0.2),

        # 第二个隐藏层,也有256个神经元
        tf.keras.layers.Dense(256,  name='h1'),
        tf.keras.layers.LeakyReLU(0.2),

        # 第三个隐藏层,也有1个神经元
        tf.keras.layers.Dense(1, name='h2'),

        # 将辨别结果映射到 [0, 1]区间，用于输出辨别结果
        # 等于1，代表输入的来句来真实样本；等于0，代表是由生成模型生成
        tf.keras.layers.Activation('sigmoid')
    ])

    return discriminator


def delete_model(model_dir='./logs/gan/model/', model_names='*.h5', max_keep=5):
    """
    为了防止模型过多占用太多的硬盘空间，删除多余的模型，只保留max_keep个模型（缺省保留5个模型）。
    参数：
        model_dir：模型所在的路径。
        model_names：模型文件名称的通配符。
        max_keep：最多保留模型的数量。缺省为最多保留5个。
    """
    model_files = glob.glob(os.path.join(model_dir, model_names))
    # 找出需要删除的模型文件列表
    del_files = sorted(model_files)[0:-max_keep]
    for file in del_files:
        os.remove(file)


def save_images(generator, epoch, test_input, image_dir='./logs/gan/image/'):
    """
    利用生成模型图片，然后，保存到指定的文件夹下。

    参数：
        generator   : 生成模型，已经经过epoch轮训练。
        epoch       : 训练的轮数。
        test_input  : 测试用的随机噪。
        image_dir   : 用于存放生成图片的路径。
    """

    predictions = generator(test_input, training=False)
    predictions = tf.reshape(predictions, [16, 28, 28, 1])

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)

        # 将生成的数据取值范围映射回到MNIST图像的像素取值范围[0, 255]
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    # 逐级创建目录，如果目录已存在，则忽略
    os.makedirs(image_dir, exist_ok=True)
    image_file_name = os.path.join(image_dir, 'image_epoch_{:04d}.png')
    plt.savefig(image_file_name.format(epoch))
    plt.close('all')  # 关闭图


# 交叉熵函数辅助类，能够评估两个数据分布之间的距离（差异，即误差）
# 用于计算生成模型、辨别模型的误差。
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_img, fake_img):
    """
    辨别模型（D)的损失函数。

    参数:
      real_img: 真实样本中的图片。
      fake_img: 由生成模型（G）生成的图片
    返回值:
      辨别模型的误差
    """

    # 识别出是真实样本图片的误差。指定真实样本的所属类别：1
    real_loss = cross_entropy(tf.ones_like(real_img), real_img)

    # 识别出是由生成图片的误差。指定生成的图片所属类别代码：0
    fake_loss = cross_entropy(tf.zeros_like(fake_img), fake_img)

    # 辨别模型的总误差
    d_loss = real_loss + fake_loss
    return d_loss


def generator_loss(fake_img):
    """
    生成模型（G）的损失函数。生成模型的目标是尽可能的欺骗辨别模型，
    让辨别模型将它生成的图片错误的识别成来源于真实样本，因此，生成模型的
    误差是按照真实样本的所属类别代码来计算的。

    参数:
      fake_img: 由生成模型（G）生成的图片
    返回:
      辨别模型的误差
    """
    # 请注意，这里必须按照真实样本所属类别来指定（tf.ones_like）
    return cross_entropy(tf.ones_like(fake_img), fake_img)


# 在TensorFlow 2.0中，@tf.function用于将模型标注为“可编译”，
# 可编译函数，用于模型编译和模型训练过程的函数
@tf.function
def train_step(images, noise, generator, g_optimizer, discriminator, d_optimizer):
    """
    完成一个批次的样本数据的训练。

    参数:
      images: 真实的样本数据。
      noise: 生成的随机噪声。
      generator: 生成模型（G）。
      g_optimizer: 生成模型优化器。
      discriminator: 辨别模型（D）。
      d_optimizer: 辨别模型优化器。
    """

    # 完成一个批次的前向传播，并计算损失。
    with tf.GradientTape() as g_type, tf.GradientTape() as d_type:
        # 生成模型根据随机噪声，生成一个批次的图片
        fake_img = generator(noise, training=True)

        # 辨别模型对真实样本数据的辨别结果
        real_output = discriminator(images, training=True)

        # 辨别模型对生成的图片数据的辨别结果
        fake_output = discriminator(fake_img, training=True)

        # 分别计算生成模型和辨别模型的损失
        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

        # 固定生成模型参数，计算辨别模型梯度，并更新辨别模型的参数
        # 从理论上来说，一般需要更新k轮辨别模型的参数，才更新一次生成模型的参数
        # 但是，在实践中，发现k=1是性能最好的方案，所以，这里每更新一次辨别模型
        # 参数，就对应的更新一次生成模型参数（k=1）
        d_gradients = d_type.gradient(
            d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(
            zip(d_gradients, discriminator.trainable_variables))

        # 固定辨别模型参数，计算生成模型梯度，并更新生成模型的参数。
        g_gradients = g_type.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(
            zip(g_gradients, generator.trainable_variables))


def train(dataset, epochs, noise_dim=100, model_dir='./logs/gan/model/',
          image_dir='./logs/gan/image/'):
    """
    GAN模型训练。自动检查是否有上一次训练过程中保存的模型，如果有自动接着上一次的模型继续训练。
    每训练10轮保存一次模型，最多保留5个保存的模型文件。

    参数：
        dataset    : 样本数据集
        epochs     : 训练的轮数。使用全部的训练数据训练一次，叫做一轮。
        noise_dim  : 随机噪声向量的长度
        model_dir  : 保存模型的路径。可以将模型保存下来，下一次从上一次保存点继续训练
        image_dir  : 用于存放生成图片的路径。
    """
    # 检查是否有上一次训练保存的模型，从上一次保存的地方开始训练
    g_files = glob.glob(os.path.join(model_dir, "generator_*.h5"))
    d_files = glob.glob(os.path.join(model_dir, "discriminator_*.h5"))

    # 起始训练轮数，自动从上一次的训练轮数继续。从0开始。
    start_epoch = 0

    # 如果上一次保存的模型存在，则读取上一次训练的模型
    if os.path.exists(model_dir) and len(g_files) > 0 and len(d_files) > 0:
        g_file = sorted(g_files)[-1]
        generator = tf.keras.models.load_model(g_file)

        d_file = sorted(d_files)[-1]
        discriminator = tf.keras.models.load_model(d_file)

        # 从上一次训练的轮数开始，继续训练
        start_epoch = int(g_file[g_file.rindex('_')+1:-3])
    else:
        # 没找到上一次训练的模型，从新创建生成模型和辨别模型
        # 构建生成模型
        generator = build_generator_model()
        # 构建辨别模型
        discriminator = build_discriminator_model()

    # 模型训练过程中，每一轮我们保存一张图片，然后，用这些图片生成动图
    # 以便于直观展示随着训练论述增加，生成的图片逐渐清晰的过程
    # 为此，我们需要一个固定的随机噪声，以便于比较对比生成图片变化的过程
    seed = tf.random.normal([16, noise_dim])

    # 优化器，辨别模型和生成模型都优化器
    g_optimizer = tf.keras.optimizers.Adam(1e-4)
    d_optimizer = tf.keras.optimizers.Adam(1e-4)

    # 生成一个批次随机噪声，用于模型训练
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    # 逐个轮次训练，每一轮将使用全部的样本数据一次
    for epoch in range(start_epoch, start_epoch+epochs):
        # 训练的开始时间点
        start = time.time()

        # 逐个批次的使用样本数据训练
        for image_batch in dataset:
            # 训练一个批次的数据
            train_step(image_batch, noise, generator, g_optimizer,
                       discriminator, d_optimizer)

        # 将本轮的训练成果保存下来，为生成动图做准备
        display.clear_output(wait=True)
        save_images(generator, epoch + 1, seed)

        # 每10轮保存一次模型
        if (epoch + 1) % 10 == 0:
            # 保存生成模型
            g_file = os.path.join(
                model_dir, "generator_{:4d}.h5".format(epoch+1))
            generator.save(g_file)

            # 保存辨别模型
            d_file = os.path.join(
                model_dir, "discriminator_{:4d}.h5".format(epoch+1))
            discriminator.save(d_file)

            # 最多只保留5个最新的模型，删除其它的模型
            delete_model(model_dir, 'generator_*.h5', 5)
            delete_model(model_dir, 'discriminator_*.h5', 5)

        # 采用空格右对齐的方式
        print('第 {: >2d} 轮，用时： {} 秒'.format(
            epoch + 1,  "{:.2f}".format(time.time()-start).rjust(6)))

    # 完成全部训练轮数，将之前所有的图片生成动图
    display.clear_output(wait=True)
    save_images(generator, epochs, seed)


# BUFFER_SIZE = 60000
# BATCH_SIZE = 256
# noise_dim = 100
# EPOCHS = 10000

BUFFER_SIZE = 60000  # 减少
BATCH_SIZE = 128  # 减小批次大小
noise_dim = 100
EPOCHS = 1000  # 减少


def gan():
    """
    模型训练的入口函数。
    """
    # 读取MNIST样本数据
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # 将MNIST样本数据，按照28×28×1的形状整理
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')

    # 将MNSIT的像素取值区间从[0, 255]映射到[-1， 1]区间，
    train_images = (train_images - 127.5) / 127.5

    # 对样本数据进行乱序排列、并按照BATCH_SIZE大小划分成不同批次数据
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # 逐级创建保存模型的目录，如果目录已存在，则忽略
    model_dir = "./logs/gan/model/"
    os.makedirs(model_dir, exist_ok=True)

    # 逐级创建保存图像的目录，如果目录已存在，则忽略
    image_dir = "./logs/gan/image/"
    os.makedirs(image_dir, exist_ok=True)

    # 模型训练
    train(train_dataset, EPOCHS, noise_dim, model_dir, image_dir)

    # 生成动画
    anim_file = os.path.join(image_dir, "gan.gif")
    with imageio.get_writer(anim_file, mode='I') as writer:
        # 读取所有的png文件列表
        filenames = glob.glob(os.path.join(image_dir, "image*.png"))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            # 将第i张图片，生成第frame帧图像，保存到gif动画中
            # 将i开根号，再翻一倍，如果比之前帧的轮数大，则加入，否则放弃
            frame = 2*(i**0.5)

            # 检查第i张图片的计算出来的序号是否比之前的轮数大
            if round(frame) > round(last):
                last = frame
            else:
                continue

            # 将png图片保存到动画中。
            image = imageio.imread(filename)
            writer.append_data(image)


# GAN模型入口函数
if __name__ == '__main__':
    gan()
