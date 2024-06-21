import numpy as np


# 论文原文：
# 顺序注入：是一种确定性的方法，它按照特定的顺序对随机向量的维度进行变异。这种方法首先从随机向量的第一个维度开始，然后逐个向后变异，直到最后一个维度。
# 在每次变异时，可以对当前维度的值应用一定的操作，如加法、减法、乘法或除法。这种策略可以帮助我们系统地探索随机向量的不同变化，了解它们对 GAN 的影响。
# 策略选择一个位置 p，1 ≤ p ≤ 100，将向量 v 中的元素向右移动一位。v′ = (v1, . . . , vp−1, vp, . . . , v100)
def sequential_injection(vector, p, operation='add', value=0.1):
    """
    顺序注入变异策略。

    参数:
    - vector: 输入向量。
    - p: 开始顺序注入的位置。
    - operation: 应用于位置p处元素的操作。选项有 'add'、'subtract'、'multiply'、'divide'。
    - value: 用于操作的值。

    返回:
    - new_vector: 变异后的向量。
    """
    new_vector = vector.copy()  # 复制输入向量
    if p < 1 or p > len(vector):
        raise ValueError("位置p必须在向量长度范围内")  # 检查位置p是否在向量范围内

    # 根据指定的操作对位置p处的元素进行变异
    if operation == 'add':
        new_vector[p - 1] += value
    elif operation == 'subtract':
        new_vector[p - 1] -= value
    elif operation == 'multiply':
        new_vector[p - 1] *= value
    elif operation == 'divide':
        if value != 0:
            new_vector[p - 1] /= value
        else:
            raise ValueError("不能除以零")
    else:
        raise ValueError("不支持的操作")

    # 将从位置p开始的元素向右移动一位
    # 例如，假设 new_vector 为 [a, b, c, d, e]，且 p = 3，则插入后的向量为 [a, b, c, c, d, e]。
    new_vector = np.insert(new_vector, p, new_vector[p - 1])
    # 例如，假设插入后的向量为 [a, b, c, c, d, e]，则去掉最后一个元素后的向量为 [a, b, c, c, d]。
    new_vector = new_vector[:-1]  # 确保向量长度不变

    return new_vector

# 论文原文：
# 随机注入 (Random injection)：是一种随机的方法，它在随机向量的维度上进行随机选择并进行变异。
# 这种方法在每次变异时，都会随机选择一个维度并对其值进行变化。与顺序注入相比，随机注入能够更广泛地探索可能的变异空间，有可能找到更具挑战性的输入。
# 然而，全随机注入可能导致一些维度被多次变异，而其他维度则未被变异。我们对随机算法进行一定的调整，保证每一个维度都变异且变异一次。
def random_injection(vector, operation='add', value=0.1):
    """
    随机注入变异策略。

    参数:
    - vector: 输入向量。
    - operation: 应用于每个维度的操作。选项有 'add'、'subtract'、'multiply'、'divide'。
    - value: 用于操作的值。

    返回:
    - new_vector: 变异后的向量。
    """
    new_vector = vector.copy()  # 复制输入向量
    indices = np.arange(len(vector))  # 生成索引数组
    np.random.shuffle(indices)  # 随机打乱索引数组

    for i in indices:
        # 根据指定的操作对每个维度进行变异
        if operation == 'add':
            new_vector[i] += value
        elif operation == 'subtract':
            new_vector[i] -= value
        elif operation == 'multiply':
            new_vector[i] *= value
        elif operation == 'divide':
            if value != 0:
                new_vector[i] /= value
            else:
                raise ValueError("不能除以零")
        else:
            raise ValueError("不支持的操作")

    return new_vector

# 数据扰动策略，包括：方向选择、步幅调整
def data_perturbation(vector, alpha=0.1):
    """
    方向选择数据扰动策略。

    参数:
    - vector: 输入向量。
    - alpha: 步幅因子。

    返回:
    - new_vector: 经过方向选择和步幅调整后的向量。
    """
    new_vector = vector.copy()  # 复制输入向量
    d = np.random.choice([-1, 0, 1], size=vector.shape)  # 随机生成扰动方向向量

    # 计算新的向量 v' + α * d
    new_vector = new_vector + alpha * d

    return new_vector


if __name__ == '__main__':
    # 测试代码。

    # 1.sequential_injection
    vector = np.random.randn(100)  # 生成随机向量
    p = np.random.randint(1, 101)  # 随机选择位置p
    print("选择的p是:", p)
    mutated_vector = sequential_injection(vector, p, operation='add', value=0.1)
    print("原始向量:", vector)
    print("变异向量:", mutated_vector)

    # 2.random_injection
    vector = np.random.randn(100)  # 生成随机向量
    mutated_vector = random_injection(vector, operation='add', value=0.1)
    print("原始向量:", vector)
    print("变异向量:", mutated_vector)

    # 3.data_perturbation
    vector = np.random.randn(100)  # 生成随机向量 该向量是是经过维度变异的向量，这里用随机向量进行测试。
    alpha = 0.1  # 设置步幅因子
    mutated_vector = data_perturbation(vector, alpha)
    print("原始向量:", vector)
    print("扰动向量:", mutated_vector)
