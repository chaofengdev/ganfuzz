#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import matplotlib.pyplot as plt
from IPython import display

import tensorflow as tf  # TF2


assert tf.__version__.startswith('2')


class GANTrainer(object):
    """
    通用的GAN模型训练器，用于只包含两个生成模型和辨别模型的GAN模型训练。

    Args:
      generator          : 创建生成模型函数。
      generator          : 创建辨别模型函数。
      generator_loss     : 生成模型损失函数。
      discriminator_loss : 辨别模型损失函数。
      generator_optimizer   : 生成模型优化器。
      discriminator_optimizer   : 辨别模型优化器。
      config: 模型训练过程中的参数配置。
      name: GAN模型的名称，比如"DCGAN"、“CGAN”等等
    """

    def __init__(self, generator, discriminator,
                 generator_loss=None,
                 discriminator_loss=None,
                 generator_optimizer=tf.keras.optimizers.Adam(1e-4),
                 discriminator_optimizer=tf.keras.optimizers.Adam(1e-4),
                 config=None,
                 noise_dim=100,
                 name="GANTrainer"):
        self.generator = generator()
        self.discriminator = discriminator()
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        # 缺省时使用的交叉熵作为损失函数（用户不指定generator_loss_fn
        # 或者 discriminator_loss_fn 时使用）
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.name = name

        # 如果用户没有指定生成模型损失函数，则创建缺省的生成模型损失函数
        if self.generator_loss is None:
            self.generator_loss = self._generator_loss

        # 如果用户没有指定辨别模型损失函数，则创建缺省的辨别模型损失函数
        if self.discriminator_loss is None:
            self.discriminator_loss = self._discriminator_loss

        # 如果不指定运行时配置对象，默认的模型保存路径'./logs/xxxx/model/'，
        # 其中，xxxx代表GAN的名称。其它参数，如模型最大保留个数等取默认值（5个）
        if config is None:
            model_dir = "./logs/{}/model/".format(name)
            os.makedirs(model_dir, exist_ok=True)
            self.config = tf.estimator.RunConfig(model_dir=model_dir)
        else:
            # 创建保存模型用的文件目录（如果已存在，则忽略）
            os.makedirs(self.config.model_dir, exist_ok=True)
        self.noise_dim = noise_dim
        self.seed = tf.random.normal([16, self.noise_dim])
        # 检查点保存函数，用于从上一次保存点继续训练
        self.checkpoint = tf.train.Checkpoint(
            # 训练的步数（全局步数，global steps）
            step=tf.Variable(1),
            # 训练的轮次
            epoch=tf.Variable(1),
            # 随机噪声张量的元素个数
            noise_dim=tf.Variable(self.noise_dim),
            # 用于测试的随机噪声的种子，该随机数保持不变
            # 用于测试经过不同轮次训练的生成模型，
            # 比较生成的图片的质量变化
            seed=tf.Variable(self.seed),
            # 生成模型、辨别模型和它们的优化器
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer
        )

    def _generator_loss(self, generated_data):
        """
        使用交叉熵计算生成模型的损失。
        """
        return self.cross_entropy(tf.ones_like(generated_data), generated_data)

    def _discriminator_loss(self, real_data, generated_data):
        """
        利用交叉熵计算辨别模型的损失。辨别模型的损失包括将样本数据识别为“真实”的损失，
        和将由生成模型生成的数据识别为“生成”的损失。
        """
        real_loss = self.cross_entropy(tf.ones_like(real_data), real_data)
        generated_loss = self.cross_entropy(
            tf.zeros_like(generated_data), generated_data)

        total_loss = real_loss + generated_loss

        return total_loss

    # tf.function标记，表示该函数将会被编译到计算图中，实现计算加速
    @tf.function
    def train_step(self, real_datas, step, batch_size=128, noise_dim=100,
                   adventage=1):
        """
        完成一个批次的样本数据训练。

        Args:
         real_datas: 一个批次的样本数据集
         batch_size: 批处理的大小
         noise_dim: 随机噪声的长度
         adventage: 为了避免生成模型坍塌，每训练生成模型adventage次，才训练辨别模型1次。

        Returns:
          生成模型的损失、辨别模型的损失。
        """
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # 调用生成模型生成样本数据
            generated_data = self.generator(noise, training=True)

            # 分别调用辨别模型识别真实的样本数据和生成的样本数据
            real_output = self.discriminator(real_datas, training=True)
            generated_output = self.discriminator(
                generated_data, training=True)

            # 分别计算生成模型和辨别模型的损失
            g_loss = self.generator_loss(generated_output)
            d_loss = self.discriminator_loss(
                real_output, generated_output)

            # 计算生成模型和辨别模型的梯度
            g_gradients = g_tape.gradient(
                g_loss, self.generator.trainable_variables)
            # 固定辨别模型参数、优化生成模型
            self.generator_optimizer.apply_gradients(zip(
                g_gradients, self.generator.trainable_variables))

            # 固定生成模型参数、优化辨别模型
            # 为了避免生成模型坍塌，每训练生成模型adventage次，才训练1次辨别模型。
            # step==1时，首次创建计算图，需要创建d_gradients变量
            if step % adventage == 0 or 'd_gradients' not in vars():
                d_gradients = d_tape.gradient(
                    d_loss, self.discriminator.trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(
                    d_gradients, self.discriminator.trainable_variables))

            return g_loss, d_loss

    def train(self, dataset, epochs=10, batch_size=128,
              show_msg_steps=2, noise_dim=100,
              adventage=1, save_checkpoints_steps=100,
              save_image_steps=1000, save_image_func=None):
        """
        GAN模型训练，共训练epochs轮次。

        Args:
            dataset: 训练数据集。
            epochs: 训练轮次
            batch_size: 每个训练批次使用的样本数量。
            noise_dim: 随机噪声张量长度。
            adventage: 为了避免生成模型坍塌，每训练生成模型adventage次，才训练辨别模型1次。
            save_checkpoints_steps: 每训练save_image_func步，保存一次模型。
            save_image_steps: 每训练save_image_steps步数，调用生成模型生成一次图片。
            save_image_func: 保存生成图片的函数。
        """
        # 检查是否有上一次训练过程中保存的模型
        manager = tf.train.CheckpointManager(
            self.checkpoint, self.config.model_dir,
            max_to_keep=self.config.keep_checkpoint_max)
        # 如果有，则加载上一次保存的模型；
        self.checkpoint.restore(manager.latest_checkpoint)

        # 检查是否加载成功
        if manager.latest_checkpoint:
            print("从上一次保存点恢复：{}\n".format(manager.latest_checkpoint))
            # 模型训练过程中，每一轮我们保存一张图片，然后，用这些图片生成动图
            # 以便于直观展示随着训练论数增加，生成的图片逐渐清晰的过程
            # 为此，我们需要一个固定的随机数种子，以便于比较对比生成图片变化的过程
            # 我们将随机噪声的种子也保存在checkpoint中，每次从checkpoint读取，确保不变
            self.seed = self.checkpoint.seed
            self.noise_dim = self.checkpoint.noise_dim
            self.generator = self.checkpoint.generator
            self.discriminator = self.checkpoint.discriminator
            self.generator_optimizer = self.checkpoint.generator_optimizer
            self.discriminator_optimizer = self.checkpoint.discriminator_optimizer
        else:
            # 使用缺省的生成模型和辨别模型（构建模型训练器时已创建,这里无需创建）
            print("重新创建模型。\n")

        time_start = time.time()

        # 从1开始计算轮数，轮数保存在checkpoint对象中，每次从上一次的轮数开始
        for epoch in range(int(self.checkpoint.epoch), epochs+1):
            # 对本轮所有的样本数据进行逐个批次的训练
            for batch_imgs in dataset:
                # 当前进行了多少个批次的训练（第几步）
                step = int(self.checkpoint.step)

                # 对本批次进行训练
                g_loss, d_loss = self.train_step(
                    batch_imgs, step, batch_size, noise_dim, adventage)
                # 训练步数加1
                self.checkpoint.step.assign_add(1)

                time_end = time.time()

                # 如果超过60秒，则输出一次日志，显示程序没有挂死
                if time_end-time_start > 60:
                    tmp = "第 {}轮， 第 {}步。生成模型损失：: {:.6f} 辨别模型损失: {:.6f}用时: {:>.2f}秒。 "
                    print(tmp.format(epoch, step, g_loss,
                                     d_loss, time_end-time_start))
                    time_start = time_end

                # 每训练save_checkpoints_steps步，保存一次模型
                if step % save_checkpoints_steps == 0:
                    save_path = manager.save()
                    tmp = "保存模型，第 {}轮， 第 {}步。用时: {:>.2f}秒。文件名: {}"
                    print(tmp.format(epoch, step,
                                     time_end-time_start, save_path))
                    time_start = time_end

                # 每训练save_image_steps步，生成一次图像，比较生成图像的变化
                if step % save_image_steps == 0 and \
                        save_image_func is not None:
                    tmp = '第 {}步, 生成模型损失： {:.2f}, 辨别模型损失： {:.2f}'
                    print(tmp.format(step, g_loss, d_loss))

                    image_dir = './logs/{}/image/'.format(self.name)
                    # 将本轮的训练成果保存下来，为生成动图做准备
                    gen_samples = self.generator(self.seed, training=False)
                    # 保存图片
                    save_image_func(gen_samples, step=step,
                                    image_dir=image_dir)

            # 完成一轮训练，轮次增加1次
            self.checkpoint.epoch.assign_add(1)
