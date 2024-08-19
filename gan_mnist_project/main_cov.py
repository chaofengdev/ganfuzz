import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os

from gan_mnist_project.mutation.mutation_strategy import data_perturbation, sequential_injection, random_injection
from models.gan import make_generator_model
from mutation.mutation_algorithms import gaussian_mutation


def calculate_confidence(classifier, generator, image):
    """
    计算生成图片的分类器置信度。

    Args:
        classifier (tf.keras.Model): 分类器模型。
        generator (tf.keras.Model): 生成器模型。
        image (tf.Tensor): 生成的图片张量。

    Returns:
        float: 分类器预测的最大置信度。
    """
    prediction = classifier.predict(image)
    return np.max(prediction)


def calculate_coverage(generator, input_vector):
    """
    计算生成器模型生成图片的神经元覆盖率。

    Args:
        generator (tf.keras.Model): 生成器模型。
        input_vector (np.ndarray): 输入向量，用于生成图片。

    Returns:
        float: 生成图片在生成器模型中激活的神经元比例。
    """
    generated_image = generator(np.expand_dims(input_vector, axis=0), training=False)
    activations = generated_image.numpy() > 0.0
    covered_neurons = np.mean(activations)
    return covered_neurons


def main(use_coverage=True):
    # 确保使用 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 设置 GPU 内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # 加载已有的MNIST分类器
    classifier_path = 'models/classifier_model/mnist_classifier.h5'
    classifier = tf.keras.models.load_model(classifier_path)
    print("分类器加载完毕")

    # 加载最新的GAN模型
    generator = make_generator_model()
    latest_checkpoint = tf.train.latest_checkpoint('models/logs/dcgan/model/')
    if latest_checkpoint:
        checkpoint = tf.train.Checkpoint(generator=generator)
        checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"已加载最新的GAN模型：{latest_checkpoint}")
    else:
        print("未找到任何GAN模型检查点文件")

    # 创建种子向量队列
    seed_queue = [tf.random.normal([100]) for _ in range(20)]

    # 初始化平均置信度或覆盖率
    if use_coverage:
        average_metric = np.mean([calculate_coverage(generator, vec) for vec in seed_queue])
    else:
        average_metric = np.mean(
            [calculate_confidence(classifier, generator, np.expand_dims(vec, axis=0)) for vec in seed_queue])

    print(f"初始时，平均{'覆盖率' if use_coverage else '置信度'}：{average_metric}")
    print("种子向量队列创建完毕")

    # 主循环
    for iteration in range(500):
        print(f"迭代 {iteration + 1}/500 开始")
        # 随机选择一个向量并进行变异
        seed_vector = random.choice(seed_queue)
        print(f"选中的种子向量: {seed_vector}")

        # 随机选择一种变异策略
        strategy = random.choice(['sequential', 'random', 'perturbation'])
        if strategy == 'sequential':
            p = np.random.randint(1, 101)  # 随机选择位置p
            mutated_vector = sequential_injection(seed_vector.numpy(), p, operation='add', value=0.1)
        elif strategy == 'random':
            mutated_vector = random_injection(seed_vector.numpy(), operation='add', value=0.1)
        elif strategy == 'perturbation':
            mutated_vector = data_perturbation(seed_vector.numpy(), alpha=0.1)
        mutated_vector = tf.convert_to_tensor(mutated_vector)  # 转换回Tensor
        print(f"变异后的向量: {mutated_vector}")

        # 生成图片 将向量送进gan中生成图片
        generated_image = generator(np.expand_dims(mutated_vector, axis=0), training=False)
        print(f"生成的图片 shape: {generated_image.shape}")

        # 计算当前迭代的指标（覆盖率或置信度）
        if use_coverage:
            metric = calculate_coverage(generator, mutated_vector)
        else:
            metric = calculate_confidence(classifier, generator, generated_image)

        print(f"当前{'覆盖率' if use_coverage else '置信度'}: {metric}")

        # 更新种子队列
        if metric < average_metric:
            seed_queue.append(mutated_vector)
            average_metric = np.mean([calculate_coverage(generator, vec) if use_coverage else calculate_confidence(
                classifier, generator, np.expand_dims(vec, axis=0)) for vec in seed_queue])
            print(f"更新后的平均{'覆盖率' if use_coverage else '置信度'}: {average_metric}")

        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Average {'Coverage' if use_coverage else 'Confidence'}: {average_metric}")

        if average_metric < 0.2:
            print("平均{'覆盖率' if use_coverage else '置信度'}低于0.2，停止迭代")
            break

    # 保存生成的图片 将队列中最后20个向量，输入到gan中得到图片。
    final_images = [generator(np.expand_dims(vec, axis=0), training=False) for vec in seed_queue[-20:]]
    final_images = np.array(final_images)

    # 保存图片并命名
    if not os.path.exists('save'):
        os.makedirs('save')

    for i, image in enumerate(final_images):
        plt.imshow(image[0, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig(f'save/generated_image_{i + 1}.png', bbox_inches='tight')

    print("生成的图片已保存")


if __name__ == '__main__':
    # 在此切换使用覆盖率或置信度计算
    main(use_coverage=True)
    # main(use_coverage=False)
