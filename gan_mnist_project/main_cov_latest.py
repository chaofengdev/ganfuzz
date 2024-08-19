import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os

from gan_mnist_project.mutation.mutation_strategy import data_perturbation, sequential_injection, random_injection
from models.gan import make_generator_model
from mutation.mutation_algorithms import gaussian_mutation
from my_cov.metrics import nac  # 导入nac类，用于神经元覆盖率的计算


# 函数1: 基于置信度的检测
def confidence_based_detection(generator, classifier, seed_queue, average_confidence, iteration_limit=500):
    """
    基于置信度的检测函数。
    对种子向量进行变异，通过生成器生成图片，使用分类器计算置信度，
    并根据置信度更新种子队列。
    """
    for iteration in range(iteration_limit):
        print(f"迭代 {iteration + 1}/{iteration_limit} 开始")
        seed_vector = random.choice(seed_queue)  # 从队列中随机选择一个种子向量
        print(f"选中的种子向量: {seed_vector}")

        # 随机选择一种变异策略并应用于选中的种子向量
        strategy = random.choice(['sequential', 'random', 'perturbation'])
        if strategy == 'sequential':
            p = np.random.randint(1, 101)  # 随机选择位置p
            mutated_vector = sequential_injection(seed_vector.numpy(), p, operation='add', value=0.1)
        elif strategy == 'random':
            mutated_vector = random_injection(seed_vector.numpy(), operation='add', value=0.1)
        elif strategy == 'perturbation':
            mutated_vector = data_perturbation(seed_vector.numpy(), alpha=0.1)
        mutated_vector = tf.convert_to_tensor(mutated_vector)  # 将变异后的向量转换为Tensor
        print(f"变异后的向量: {mutated_vector}")

        # 使用生成器生成图片
        generated_image = generator(np.expand_dims(mutated_vector, axis=0), training=False)
        print(f"生成的图片 shape: {generated_image.shape}")

        # 使用分类器预测生成图片的标签和置信度
        prediction = classifier.predict(generated_image)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        print(f"预测标签: {predicted_label}, 置信度: {confidence}")

        # 如果置信度低于平均值，将变异后的向量加入种子队列，并更新平均置信度
        if confidence < average_confidence:
            seed_queue.append(mutated_vector)
            average_confidence = np.mean(
                [np.max(classifier.predict(generator(np.expand_dims(vec, axis=0), training=False))) for vec in seed_queue]
            )
            print(f"更新后的平均置信度: {average_confidence}")

        # 每100次迭代打印一次当前状态
        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Average Confidence: {average_confidence}")

        # 如果平均置信度低于0.2，停止迭代
        if average_confidence < 0.2:
            print("平均置信度低于0.2，停止迭代")
            break

    return seed_queue  # 返回更新后的种子队列


# 函数2: 基于神经元覆盖率的检测
def coverage_based_detection(generator, seed_queue, layers, iteration_limit=500, threshold=0):
    """
    通过变异和筛选来优化生成器的神经元覆盖率。

    参数:
    - generator: GAN生成器模型
    - seed_queue: 初始随机向量的种子队列
    - layers: 生成器模型的层列表，用于计算神经元覆盖率
    - iteration_limit: 迭代次数上限
    - threshold: 神经元激活的阈值

    返回:
    - seed_queue: 更新后的种子队列，包含高神经元覆盖率的向量
    """

    def calculate_coverage(vec):
        """
        计算一个随机向量在生成器中的神经元覆盖率
        """
        coverage = []
        for layer in layers:
            intermediate_layer_model = tf.keras.Model(inputs=generator.input, outputs=layer.output)
            intermediate_output = intermediate_layer_model(vec)
            if len(intermediate_output.shape) > 2:
                intermediate_output = tf.reduce_mean(intermediate_output, axis=[1, 2])  # 平均化高维激活
            coverage.append(np.sum(intermediate_output.numpy() > threshold, axis=1))  # 计算激活神经元数量
        return np.mean(coverage)  # 返回覆盖率的平均值

    def rank_2(seed_queue):
        """
        对种子队列中的随机向量按照神经元覆盖率进行排序
        """
        coverage_scores = [(vec, calculate_coverage(vec)) for vec in seed_queue]
        coverage_scores.sort(key=lambda x: x[1], reverse=True)
        return [vec for vec, _ in coverage_scores]

    # 2. 计算初始种子队列的平均神经元覆盖率
    initial_coverage = [calculate_coverage(tf.expand_dims(vec, axis=0)) for vec in seed_queue]
    average_coverage = np.mean(initial_coverage)
    print(f"初始时，平均神经元覆盖率：{average_coverage}")

    for iteration in range(iteration_limit):
        print(f"迭代 {iteration + 1}/{iteration_limit} 开始")

        # 3. 从种子队列中随机选择一个向量
        seed_vector = random.choice(seed_queue)
        print(f"选中的种子向量: {seed_vector}")

        # 4. 对选中的随机向量进行变异
        strategy = random.choice(['sequential', 'random', 'perturbation'])
        if strategy == 'sequential':
            p = np.random.randint(1, 101)
            mutated_vector = sequential_injection(seed_vector.numpy(), p, operation='add', value=0.1)
        elif strategy == 'random':
            mutated_vector = random_injection(seed_vector.numpy(), operation='add', value=0.1)
        elif strategy == 'perturbation':
            mutated_vector = data_perturbation(seed_vector.numpy(), alpha=0.1)
        mutated_vector = tf.convert_to_tensor(mutated_vector)
        print(f"变异后的向量: {mutated_vector}")

        # 5. 将变异后的向量传入生成器，计算神经元覆盖率
        mutated_coverage = calculate_coverage(tf.expand_dims(mutated_vector, axis=0))
        print(f"变异后向量的神经元覆盖率：{mutated_coverage}")

        # 6. 如果神经元覆盖率增加，将变异后的向量加入种子队列
        if mutated_coverage > average_coverage:
            seed_queue.append(mutated_vector)
            # 更新平均覆盖率
            average_coverage = np.mean([calculate_coverage(tf.expand_dims(vec, axis=0)) for vec in seed_queue])
            print(f"更新后的平均神经元覆盖率：{average_coverage}")

        # 打印每100次迭代的状态
        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Average Coverage: {average_coverage}")

    # 7. 使用 rank_2 方法对种子队列进行排序
    sorted_seed_queue = rank_2(seed_queue)

    return sorted_seed_queue  # 返回更新后的种子队列


# 函数2: 基于神经元覆盖率的检测 修改代码
def coverage_based_detection2(generator, seed_queue, layers, iteration_limit=50, threshold=0):
    """
    通过变异和筛选来优化生成器的神经元覆盖率。

    参数:
    - generator: GAN生成器模型
    - seed_queue: 初始随机向量的种子队列
    - layers: 生成器模型的层列表，用于计算神经元覆盖率
    - iteration_limit: 迭代次数上限
    - threshold: 神经元激活的阈值

    返回:
    - seed_queue: 更新后的种子队列，包含高神经元覆盖率的向量
    """

    def calculate_coverage(vec):
        """
        计算一个随机向量在生成器中的神经元覆盖率。这里的代码还需要完善。
        """
        total_neurons = 0
        activated_neurons = 0
        for layer in layers:
            intermediate_layer_model = tf.keras.Model(inputs=generator.input, outputs=layer.output)
            intermediate_output = intermediate_layer_model(vec)
            if len(intermediate_output.shape) > 2:  # 对卷积层输出进行均值化
                intermediate_output = tf.reduce_mean(intermediate_output, axis=[1, 2])

            # 计算激活的神经元数量和总神经元数量
            activated_neurons += np.sum(intermediate_output.numpy() > threshold)
            total_neurons += intermediate_output.shape[-1]  # 最后一维表示神经元数量

        coverage = activated_neurons / total_neurons if total_neurons > 0 else 0
        return coverage

    def rank_2(seed_queue):
        """
        对种子队列中的随机向量按照神经元覆盖率进行排序
        """
        coverage_scores = [(vec, calculate_coverage(tf.expand_dims(vec, axis=0))) for vec in seed_queue]
        coverage_scores.sort(key=lambda x: x[1], reverse=True)
        return [vec for vec, _ in coverage_scores]

    # 2. 计算初始种子队列的平均神经元覆盖率
    initial_coverage = [calculate_coverage(tf.expand_dims(vec, axis=0)) for vec in seed_queue]
    average_coverage = np.mean(initial_coverage)
    print(f"初始时，平均神经元覆盖率：{average_coverage}")

    for iteration in range(iteration_limit):
        print(f"迭代 {iteration + 1}/{iteration_limit} 开始")

        # 3. 从种子队列中随机选择一个向量
        seed_vector = random.choice(seed_queue)
        print(f"选中的种子向量: {seed_vector}")

        # 4. 对选中的随机向量进行变异
        strategy = random.choice(['sequential', 'random', 'perturbation'])
        if strategy == 'sequential':
            p = np.random.randint(1, 101)
            mutated_vector = sequential_injection(seed_vector.numpy(), p, operation='add', value=0.1)
        elif strategy == 'random':
            mutated_vector = random_injection(seed_vector.numpy(), operation='add', value=0.1)
        elif strategy == 'perturbation':
            mutated_vector = data_perturbation(seed_vector.numpy(), alpha=0.1)
        mutated_vector = tf.convert_to_tensor(mutated_vector)
        print(f"变异后的向量: {mutated_vector}")

        # 5. 将变异后的向量传入生成器，计算神经元覆盖率
        mutated_coverage = calculate_coverage(tf.expand_dims(mutated_vector, axis=0))
        print(f"变异后向量的神经元覆盖率：{mutated_coverage}")

        # 6. 如果神经元覆盖率增加，将变异后的向量加入种子队列
        if mutated_coverage > average_coverage:
            seed_queue.append(mutated_vector)
            # 更新平均覆盖率
            average_coverage = np.mean([calculate_coverage(tf.expand_dims(vec, axis=0)) for vec in seed_queue])
            print(f"更新后的平均神经元覆盖率：{average_coverage}")

        # 打印每100次迭代的状态
        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Average Coverage: {average_coverage}")

    # 7. 使用 rank_2 方法对种子队列进行排序
    sorted_seed_queue = rank_2(seed_queue)

    # 8. 将高覆盖率向量输入生成器，生成并保存对应的图像
    # for idx, vec in enumerate(sorted_seed_queue):
    #     generated_image = generator(tf.expand_dims(vec, axis=0), training=False)
    #     plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(f'save/generated_image_{idx + 1}.png', bbox_inches='tight')
    #
    # print("生成的图片已保存")

    return sorted_seed_queue  # 返回更新后的种子队列


# 函数3: 生成并保存最终图片
def generate_and_save_images(generator, seed_queue, save_dir='save'):
    """
    生成并保存种子队列中最后20个向量对应的图片。

    参数:
    - generator: GAN生成器模型
    - seed_queue: 存储种子向量的队列
    - save_dir: 保存图片的目录
    """
    # 生成最后20个向量对应的图片
    final_images = [generator(np.expand_dims(vec, axis=0), training=False) for vec in seed_queue[-20:]]
    final_images = np.array(final_images)

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存生成的图片
    for i, image in enumerate(final_images):
        plt.imshow(image[0, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig(f'{save_dir}/generated_image_{i + 1}.png', bbox_inches='tight')

    print("生成的图片已保存")


# 主程序
def main(use_confidence=True):
    """
    主程序函数，根据输入参数选择使用置信度检测或神经元覆盖率检测。
    """
    # 确保使用 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    if use_confidence:
        # 使用置信度检测
        average_confidence = np.mean(
            [np.max(classifier.predict(generator(np.expand_dims(vec, axis=0), training=False))) for vec in seed_queue]
        )
        print(f"初始时，平均置信度：{average_confidence}")
        seed_queue = confidence_based_detection(generator, classifier, seed_queue, average_confidence)
    else:
        # 使用神经元覆盖率检测
        layers = [layer for layer in generator.layers if 'conv' in layer.name or 'dense' in layer.name]

        # 调用 coverage_based_detection 函数进行检测
        seed_queue = coverage_based_detection2(generator, seed_queue, layers)

    # 调用生成并保存图片的函数
    generate_and_save_images(generator, seed_queue, save_dir='save')


if __name__ == '__main__':
    main(use_confidence=False)  # 选择使用置信度，传入use_confidence=False则使用神经元覆盖率
