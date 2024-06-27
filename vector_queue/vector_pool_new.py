import numpy as np
import tensorflow as tf

from mutation.mutation_strategy import sequential_injection


def create_vector_queue(original_vector, num_variants=1000):
    """
    创建变异向量的队列。

    参数:
    - original_vector: 原始种子向量。
    - num_variants: 变异向量的数量。

    返回:
    - vector_queue: 包含变异向量的队列。
    """
    queue = [original_vector]
    for _ in range(num_variants):
        p = np.random.randint(1, len(original_vector) + 1)
        operation = np.random.choice(['add', 'subtract', 'multiply', 'divide'])
        value = np.random.uniform(0.05, 0.2)
        variant = sequential_injection(queue[-1], p, operation, value)
        queue.append(variant)
    return queue


# 暂定随机选择，这里涉及到具体的选择策略。
def select_vectors_from_queue(vector_queue, num_selections=128):
    """
    从变异向量队列中随机选择一定数量的向量。

    参数:
    - vector_queue: 变异向量的队列。
    - num_selections: 选择的向量数量。

    返回:
    - selected_vectors: 选择的向量，转换为Tensor。
    """
    selected_vectors = np.random.choice(vector_queue, size=num_selections, replace=False)
    return tf.convert_to_tensor(selected_vectors)


# 向量池（种子队列），用来替代gan生成模型的输入张量，原尺寸为 noise = tf.random.normal([batch_size, noise_dim])
# 将其封装为函数，并在使用gan生成器生成图片时调用。
def get_noise(batch_size=128, noise_dim=100):
    # 定义GAN模型输入的参数
    # batch_size = 128
    # noise_dim = 100

    # 原始随机向量 初始向量种子
    original_vector = np.random.randn(noise_dim)  # 根据正态分布生成的随机向量

    # 1. 创建包含变异向量的队列
    vector_queue = create_vector_queue(original_vector, num_variants=128)
    print("变异向量队列:")
    for vec in vector_queue:
        print(vec)

    # 2. 从队列中随机选择一定数量的种子，并转换为张量
    selected_vectors_tensor = select_vectors_from_queue(vector_queue, batch_size)
    print("选择的种子张量:", selected_vectors_tensor)

    # 验证张量形状是否正确
    assert selected_vectors_tensor.shape == (batch_size, noise_dim), "张量形状不匹配"
    print("张量形状正确:", selected_vectors_tensor.shape)

    # 返回指定形状的张量，作为gan生成器的输入。
    return selected_vectors_tensor
