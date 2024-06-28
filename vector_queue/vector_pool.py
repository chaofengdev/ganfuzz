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


def select_vectors_from_queue(vector_queue, num_selections=128, noise_dim=100):
    """
    从变异向量队列中随机选择一定数量的向量。
    参数:
    - vector_queue: 变异向量的队列。
    - num_selections: 选择的向量数量。
    返回:
    - selected_vectors: 选择的向量，转换为Tensor。
    """
    # 将 vector_queue 转换为 numpy 数组
    vector_queue_array = np.array(vector_queue)

    # 从 vector_queue_array 中随机选择 num_selections 个向量的索引
    selected_indices = np.random.choice(len(vector_queue_array), size=num_selections, replace=False)

    # 根据选定的索引获取向量
    selected_vectors = vector_queue_array[selected_indices]

    # 转换为 TensorFlow 张量并调整形状为 [num_selections, noise_dim]
    selected_vectors_tensor = tf.convert_to_tensor(selected_vectors, dtype=tf.float32)
    selected_vectors_tensor.set_shape([num_selections, noise_dim])

    return selected_vectors_tensor


# 向量池（种子队列），用来替代gan生成模型的输入张量，原尺寸为 noise = tf.random.normal([batch_size, noise_dim])
# 将其封装为函数，并在使用gan生成器生成图片时调用。
def get_noise(batch_size=128, noise_dim=100):
    original_vector = np.random.randn(noise_dim)
    vector_queue = create_vector_queue(original_vector, num_variants=128)

    selected_vectors_tensor = select_vectors_from_queue(vector_queue, num_selections=batch_size, noise_dim=noise_dim)
    assert selected_vectors_tensor.shape == (batch_size, noise_dim), "张量形状不匹配"

    return selected_vectors_tensor
