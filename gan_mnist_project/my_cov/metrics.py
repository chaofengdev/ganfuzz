# #!/usr/bin/env python2
# # -*- coding: utf-8 -*-
import time
from collections import defaultdict

import numpy as np
from keras import Model
from scipy import stats
from tqdm import tqdm


class MetricsTime(object):
    def __init__(self, t_collection=0, t_cam=0, t_ctm=0):
        self.t_collection = t_collection
        self.t_cam = t_cam
        self.t_ctm = t_ctm
        self.first_add_info_time = True
        self.use_batch_predict = False  # 是否开启批量预测

    def get_t_collection(self):
        return self.t_collection

    def get_t_cam(self):
        return self.t_cam

    def get_t_ctm(self):
        return self.t_ctm

    # 用于对给定的测试数据 test 进行预测，并根据是否启用了批量预测模式（use_batch_predict）来决定如何进行预测。
    def predict_by_batch(self, test, i):
        if self.use_batch_predict:  # 使用批量预测
            len_test = len(test)  # 获取测试数据的样本数量
            arr = np.linspace(0, len_test, num=11)  # 生成一个等间隔的数组 arr，它从 0 到 len_test 等分成 10 个区间（11 个点）
            num_arr = [int(x) for x in arr]  # 将 arr 中的每个值都转换为整数，并存储在 num_arr 中。
            temp_arr = []  # 初始化一个空列表 temp_arr，用于存储每个批次的预测结果
            for ix in range(len(num_arr) - 1):  # 分批处理数据
                start_ix = num_arr[ix]
                end_ix = num_arr[ix + 1]
                temp = i.predict(test[start_ix:end_ix])
                temp.astype(np.float32)
                temp_arr.append(temp)
            temp_arr = np.concatenate(temp_arr, axis=0)  # 合并所有批次的预测结果
        else:  # 不使用批量预测
            temp_arr = i.predict(test)
        return temp_arr


# deep gauge
class kmnc(MetricsTime):
    def __init__(self, train, input, layers, k_bins=1000, max_select_size=None, time_limit=43200):
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.k_bins = k_bins
        self.lst = []
        self.upper = []
        self.lower = []
        index_lst = []

        self.time_limit = time_limit
        self.max_select_size = max_select_size

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.upper.append(np.max(temp, axis=0))
            self.lower.append(np.min(temp, axis=0))

        self.upper = np.concatenate(self.upper, axis=0)
        self.lower = np.concatenate(self.lower, axis=0)
        self.neuron_num = self.upper.shape[0]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self, test):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        act_num = 0
        for index in range(len(self.upper)):
            bins = np.linspace(self.lower[index], self.upper[index], self.k_bins)
            act_num += len(np.unique(np.digitize(self.neuron_activate[:, index], bins)))
        return act_num / float(self.k_bins * self.neuron_num)

    def get_big_bins(self, test):
        s = time.time()
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        big_bins = np.zeros((len(test), self.neuron_num, self.k_bins + 1))
        for n_index, neuron_activate in tqdm(enumerate(self.neuron_activate)):
            for index in range(len(neuron_activate)):
                bins = np.linspace(self.lower[index], self.upper[index], self.k_bins)
                temp = np.digitize(neuron_activate[index], bins)
                big_bins[n_index][index][temp] = 1

        big_bins = big_bins.astype('int')
        e = time.time()
        if self.first_add_info_time:
            self.t_collection += e - s
            self.first_add_info_time = False
        return big_bins

    def rank_fast(self, test):
        big_bins = self.get_big_bins(test)
        start = time.time()
        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))
        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover_num = (big_bins[initial] > 0).sum()
        cover_last = big_bins[initial]
        while True:
            end = time.time()
            if self.max_select_size is not None and len(subset) == self.max_select_size:
                print("max_select_size", len(subset))
                self.t_cam = end - start
                return subset
            if end - start >= self.time_limit:
                print("=======================time limit=======================")
                return subset
            flag = False
            for index in lst:
                if index in no_idx_arr:
                    continue
                temp1 = np.bitwise_or(cover_last, big_bins[index])
                now_cover_num = (temp1 > 0).sum()
                if now_cover_num > max_cover_num:
                    max_cover_num = now_cover_num
                    max_index = index
                    max_cover = temp1
                    flag = True
            cover_last = max_cover
            if not flag or len(lst) == 1:
                break
            no_idx_arr.append(max_index)
            subset.append(max_index)
            print(len(subset), end - start)
        self.t_cam = end - start
        return subset

    def rank_greedy(self, test):
        big_bins = self.get_big_bins(test)
        start = time.time()
        subset = []
        lst = list(range(len(test)))

        np.random.seed(0)

        initial = np.random.permutation(len(test))[0]
        subset.append(initial)

        max_cover_num = (big_bins[initial] > 0).sum()
        cover_last = big_bins[initial]

        for index in lst:
            if index == initial:
                continue
            temp1 = np.bitwise_or(cover_last, big_bins[index])
            now_cover_num = (temp1 > 0).sum()
            if now_cover_num > max_cover_num:
                max_cover_num = now_cover_num
                cover_last = temp1
                subset.append(index)  #
        end = time.time()
        self.t_cam = end - start

        return subset


class nbc(MetricsTime):
    def __init__(self, train, input, layers, std=0):
        super().__init__()
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.std = std
        self.lst = []
        self.upper = []
        self.lower = []
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = self.predict_by_batch(train, i).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(train, i, l.shape[-1])
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.upper.append(np.max(temp, axis=0) + std * np.std(temp, axis=0))
            self.lower.append(np.min(temp, axis=0) - std * np.std(temp, axis=0))
        self.upper = np.concatenate(self.upper, axis=0)
        self.lower = np.concatenate(self.lower, axis=0)
        self.neuron_num = self.upper.shape[0]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self, test, use_lower=False):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)

        act_num = 0
        act_num += (np.sum(self.neuron_activate > self.upper, axis=0) > 0).sum()
        if use_lower:
            act_num += (np.sum(self.neuron_activate < self.lower, axis=0) > 0).sum()

        if use_lower:
            return act_num / (2 * float(self.neuron_num))
        else:
            return act_num / float(self.neuron_num)

    def get_lower_and_upper_flag(self, test, use_lower):
        s = time.time()
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)

        upper = (self.neuron_activate > self.upper)
        if use_lower:
            lower = (self.neuron_activate < self.lower)
        else:
            lower = []

        e = time.time()
        if self.first_add_info_time:
            self.t_collection += e - s
            self.first_add_info_time = False
        return upper, lower

    def rank_fast(self, test, use_lower=False):
        upper, lower = self.get_lower_and_upper_flag(test, use_lower)
        s = time.time()
        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))

        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover_num = np.sum(upper[initial])
        if use_lower:
            max_cover_num += np.sum(lower[initial])
        cover_last_1 = upper[initial]
        if use_lower:
            cover_last_2 = lower[initial]
        while True:
            flag = False
            for index in lst:
                if index in no_idx_arr:
                    continue
                temp1 = np.bitwise_or(cover_last_1, upper[index])
                cover1 = np.sum(temp1)
                if use_lower:
                    temp2 = np.bitwise_or(cover_last_2, lower[index])
                    cover1 += np.sum(temp2)
                if cover1 > max_cover_num:
                    max_cover_num = cover1
                    max_index = index
                    flag = True
                    max_cover1 = temp1
                    if use_lower:
                        max_cover2 = temp2
            if not flag or len(lst) == 1:
                break
            no_idx_arr.append(max_index)
            # lst.remove(max_index)
            subset.append(max_index)
            cover_last_1 = max_cover1
            if use_lower:
                cover_last_2 = max_cover2
            # print(max_cover_num)
        e = time.time()
        self.t_cam = e - s
        return subset

    def rank_2(self, test, use_lower=False):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        if use_lower:
            s = time.time()
            res = np.argsort(
                np.sum(self.neuron_activate > self.upper, axis=1) + np.sum(self.neuron_activate < self.lower, axis=1))[
                  ::-1]
            e = time.time()
        else:
            s = time.time()
            res = np.argsort(np.sum(self.neuron_activate > self.upper, axis=1))[::-1]
            e = time.time()
        self.t_ctm = e - s
        return res


class tknc(MetricsTime):
    def __init__(self, test, input, layers, k=2):
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = test
        self.input = input
        self.layers = layers
        self.k = k
        self.lst = []
        self.neuron_activate = []
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = self.predict_by_batch(test, i).reshape(len(test), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, i, l.shape[-1])
            if index == 'dense':
                temp = i.predict(test).reshape(len(test), l.shape[-1])
            self.neuron_activate.append(temp)
        self.neuron_num = np.concatenate(self.neuron_activate, axis=1).shape[-1]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self, choice_index):
        neuron_activate = 0
        for neu in self.neuron_activate:
            temp = neu[choice_index]
            neuron_activate += len(np.unique(np.argsort(temp, axis=1)[:, -self.k:]))
        return neuron_activate / float(self.neuron_num)

    def get_top_k_neuron(self):
        s = time.time()
        neuron = []
        layers_num = 0
        for neu in self.neuron_activate:
            neuron.append(np.argsort(neu, axis=1)[:, -self.k:] + layers_num)
            layers_num += neu.shape[-1]
        neuron = np.concatenate(neuron, axis=1)
        e = time.time()
        self.t_collection += e - s
        return neuron

    def rank(self, test):
        neuron = self.get_top_k_neuron()

        s = time.time()
        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))

        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover = len(np.unique(neuron[initial]))

        cover_now = neuron[initial]

        while True:
            flag = False
            for index in lst:
                if index in no_idx_arr:
                    continue
                temp = np.union1d(cover_now, neuron[index])
                cover1 = len(temp)
                if cover1 > max_cover:
                    max_cover = cover1
                    max_index = index
                    flag = True
                    max_cover_now = temp
            if not flag or len(lst) == 1:
                break
            no_idx_arr.append(max_index)
            subset.append(max_index)
            cover_now = max_cover_now
        e = time.time()
        self.t_cam = e - s
        return subset


# deepxplore
class nac(MetricsTime):
    def __init__(self, test, input, layers, t=0):
        super().__init__()
        s = time.time()  # 记录开始的时间s
        self.train = test  # 测试数据
        self.input = input  # 输入张量
        self.layers = layers  # 模型中的层
        self.t = t  # 神经元激活的阈值
        self.lst = []  # 存储神经网络模型的子模型
        self.neuron_activate = []  # 存储神经元的激活值

        index_lst = []  # 局部变量 在遍历层时存储层的索引或名称

        # 这个循环用于遍历所有的层，逐层计算神经元的激活值。
        for index, l in layers:  # l是当前遍历的层。是Keras 层对象。
            self.lst.append(Model(inputs=input, outputs=l))  # 创建子模型并存储
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)  # 模型实例
            if index == 'conv':  # 是否是卷积层
                # temp = i.predict(test).reshape(len(test), -1, l.shape[-1])
                temp = self.predict_by_batch(test, i).reshape(len(test), -1, l.shape[-1])  # 预测结果重塑为形状为 (样本数, ?, 通道数) 的张量。
                temp = np.mean(temp, axis=1)  # 计算每个样本在每个通道上的平均激活值，得到形状为 (样本数, 通道数) 的张量。
                # self.predict_by_batch2(test, i, l.shape[-1])
            if index == 'dense':  # 是否是全连接层
                temp = i.predict(test).reshape(len(test), l.shape[-1])  # 直接计算测试数据的激活值，并将其重塑为形状为 (样本数, 神经元数) 的张量。
            temp = 1 / (1 + np.exp(-temp))  # 对每个激活值应用 Sigmoid 函数，将其限制在 0 到 1 之间。
            self.neuron_activate.append(temp.copy())  # 将当前层的激活值复制并添加到 self.neuron_activate 列表中。
        self.neuron_num = np.concatenate(self.neuron_activate, axis=1).shape[-1]  # 整个网络中的总神经元数
        self.lst = list(zip(index_lst, self.lst))  # 将索引（或名称）与对应的子模型配对，存储在 self.lst 中
        e = time.time()  # 记录结束时间e
        self.t_collection += e - s  # 计算从开始到结束所花费的时间，并将其累加到 self.t_collection 中。

    # 计算神经网络中神经元激活覆盖率
    def fit(self):
        neuron_activate = 0  # 局部变量，用来累计超过阈值的神经元数目
        for neu in self.neuron_activate:
            neuron_activate += np.sum(np.sum(neu > self.t, axis=0) > 0)  # 至少有一个样本下神经元激活值超过阈值t，则计数。
        return neuron_activate / float(self.neuron_num)  # 计算神经元激活覆盖率，表示有多少比例的神经元在测试数据中被激活。

    # 计算给定测试数据集 test 在神经网络各层上的神经元激活情况，并根据预定义的阈值 t 生成一个表示神经元激活情况的布尔矩阵 upper
    # 该矩阵指示哪些神经元在至少一个样本中超过了阈值。这里的逻辑和前面的init高度相似，只是为了输出upper矩阵，便于后续使用。
    def get_upper(self, test):
        self.neuron_activate = []  # 初始化空列表，存储所有层的神经元激活值。
        for index, l in self.lst:  # 遍历所有的子模型，index表示当前层的类型，l表示当前层的模型对象。
            if index == 'conv':  # 卷积层。
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)  # 样本在当前卷积层中的平均激活值。
                # self.predict_by_batch2(test, l, l.output.shape[-1])
            if index == 'dense':  # 全连接层
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])  # 直接预测，并重塑形状（样本数，神经元数）
            temp = 1 / (1 + np.exp(-temp))  # 对当前层的激活值应用 Sigmoid 函数，将其标准化为 0 到 1 之间的值。
            self.neuron_activate.append(temp.copy())  # 复制并添加到列表中，逐层存储神经元激活值。
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)  # 合并所有层的激活值，(样本数, 总神经元数)
        s = time.time()
        upper = (self.neuron_activate > self.t)  # 比较激活值与阈值t，生成bool矩阵。
        e = time.time()
        self.t_collection += e - s  # 计算激活值的时间开销。
        return upper

    # 基于贪心算法为测试数据集中的样本生成一个子集，该子集中的样本可以覆盖尽可能多的神经元激活情况。
    def rank_fast(self, test):
        upper = self.get_upper(test)  # 获得神经元激活bool矩阵upper，应该是(样本数, 总神经元数) upper[i,j]表示第i个样本的第j个神经元
        s = time.time()

        subset = []  # 存储子集样本索引
        no_idx_arr = []  # 记录已经选择过的样本索引，以避免重复选择。
        lst = list(range(len(test)))  # 测试数据集的所有样本索引。
        initial = np.random.choice(range(len(test)))  # 随机选择一个样本作为初始样本的索引。

        # lst.remove(initial)
        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover_num = np.sum(upper[initial])  # 初始样本能够覆盖的神经元数量。
        cover_last_1 = upper[initial]  # 初始样本的神经元激活情况
        while True:
            flag = False
            for index in lst:  # 测试数据集中的所有样本索引。
                if index in no_idx_arr:  # 选择过，则越过。
                    continue
                # 模拟一下，如果把当前样本 index 加入到已有的样本集里，神经元的整体覆盖情况会如何变化。
                temp1 = np.bitwise_or(cover_last_1, upper[index])  # 两个布尔数组，按位或。
                cover1 = np.sum(temp1)  # 激活的神经元个数。
                if cover1 > max_cover_num:  # 新样本能够增加整体覆盖率。
                    max_cover_num = cover1  # 更新最大覆盖率。
                    max_index = index  # 记录样本索引
                    flag = True  # 标记找到了合适的样本。
                    max_cover1 = temp1  # 记录按位或结果。
            if not flag or len(lst) == 1:  # 本轮循环中没有找到能增加覆盖率的样本、样本列表中只剩一个样本，停止循环
                break
            no_idx_arr.append(max_index)  # 样本索引加入列表
            subset.append(max_index)  # 样本索引加入加入列表
            cover_last_1 = max_cover1  # 记录当前覆盖情况
            # print(max_cover_num)
        e = time.time()
        self.t_cam = e - s  # 整个排序过程所花费的时间
        return subset  # 选择的样本子集

    # 对给定的测试数据集 test 中的样本进行排序，排序的依据是每个样本在神经网络中激活的神经元数量。
    def rank_2(self, test):
        self.neuron_activate = []  # 设置为空列表，存储神经网络中每一层的神经元激活值
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])  # (样本数, ?, 通道数)
                temp = np.mean(temp, axis=1)  # (样本数, 通道数)
                # self.predict_by_batch2(test, l, l.output.shape[-1])
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])  # (样本数, 神经元数)
            temp = 1 / (1 + np.exp(-temp))
            self.neuron_activate.append(temp.copy())  # 神经元激活列表
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)  # 合并所有层的激活值，完整的激活值矩阵(样本数, 总神经元数)

        s = time.time()
        # 1.对于每个样本，计算它激活的神经元数
        # 2.np.argsort(...)根据每个样本激活的神经元数进行排序，返回排序后的样本索引
        # 3.[::-1]将排序结果反转，使得激活神经元最多的样本排在最前面
        res = np.argsort(np.sum(self.neuron_activate > self.t, axis=1))[::-1]
        e = time.time()
        self.t_ctm = e - s  # 整个排序过程所花费的时间
        return res  # 按激活神经元数量从多到少排序后的样本索引列表 res


## deepgini
def deep_metric(pred_test_prob):
    metrics = np.sum(pred_test_prob ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    return rank_lst


def deep_metric2(pred_test_prob):
    metrics = np.sum(pred_test_prob ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    return rank_lst, 1 - metrics


# Surprise Adequacy
class LSC(MetricsTime):
    def __init__(self, train, label, input, layers, u=2000, k_bins=1000, threshold=None):

        """
        """
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.lst = []
        self.neuron_activate_train = []
        self.u = u
        self.k_bins = k_bins
        self.threshold = threshold
        self.test_score = []
        self.train_label = np.array(label)
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            temp = None
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.neuron_activate_train.append(temp.copy())  # len(train), l.shape[-1]
        self.neuron_activate_train = np.concatenate(self.neuron_activate_train, axis=1)  #
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

        self.class_matrix = None
        self._init_class_matrix()
        self.kdes, self.removed_cols = self._init_kdes()

    def _init_class_matrix(self):
        class_matrix = {}
        for i, lb in enumerate(self.train_label):
            if lb not in class_matrix:
                class_matrix[lb] = []
            class_matrix[lb].append(i)
        self.class_matrix = class_matrix

    def _init_kdes(self):
        class_matrix = self.class_matrix
        train_ats = self.neuron_activate_train
        num_classes = np.unique(self.train_label)
        removed_cols = []
        if self.threshold is not None:
            for lb in num_classes:
                col_vectors = np.transpose(train_ats[class_matrix[lb]])
                for i in range(col_vectors.shape[0]):
                    if (
                            np.var(col_vectors[i]) < self.threshold
                            and i not in removed_cols
                    ):
                        removed_cols.append(i)

        kdes = {}
        for lb in num_classes:
            refined_ats = np.transpose(train_ats[class_matrix[lb]])
            refined_ats = np.delete(refined_ats, removed_cols, axis=0)
            if refined_ats.shape[0] == 0:
                print("warning....  remove all")
                continue
            kdes[lb] = stats.gaussian_kde(refined_ats)
        return kdes, removed_cols

    def _get_lsa(self, kde, at, removed_cols):
        refined_at = np.delete(at, removed_cols, axis=0)
        return np.asscalar(-kde.logpdf(np.transpose(refined_at)))

    def fit(self, test, label):
        s = time.time()
        # print("LSC fit")
        self.neuron_activate_test = []
        self.test_score = []
        for index, l in self.lst:
            temp = None
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test = np.concatenate(self.neuron_activate_test, axis=1)  # 10000 10

        class_matrix = self._init_class_matrix()

        kdes, removed_cols = self.kdes, self.removed_cols

        for test_sample, label_sample in tqdm(zip(self.neuron_activate_test, label)):
            if label_sample in kdes.keys():
                kde = kdes[label_sample]
                self.test_score.append(self._get_lsa(kde, test_sample, removed_cols))
            else:
                self.test_score.append(0)
        e = time.time()
        self.t_collection += e - s
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        x = np.unique(np.digitize(self.test_score, bins))
        rate = len(np.unique(x)) / float(self.k_bins)
        return rate

    def get_sore(self):
        return self.test_score

    def get_u(self):
        return self.u

    def rank_fast(self):
        s = time.time()
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        score_bin = np.digitize(self.test_score, bins)
        score_bin_uni = np.unique(score_bin)
        res_idx_arr = []
        for x in score_bin_uni:
            np.random.seed(41)
            idx_arr = np.argwhere(score_bin == x).flatten()
            idx = np.random.choice(idx_arr)
            res_idx_arr.append(idx)
        # print(len(res_idx_arr), self.k_bins * self.get_rate())
        e = time.time()
        self.t_cam = e - s
        return res_idx_arr

    def rank_2(self):
        s = time.time()
        res = np.argsort(self.get_sore())[::-1]
        e = time.time()
        self.t_ctm = e - s
        return res


## DSC
class DSC(MetricsTime):
    def __init__(self, train, label, input, layers, u=2, k_bins=1000, threshold=10 ** -5, time_limit=3600):
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.lst = []
        self.std_lst = []
        self.mask = []
        self.neuron_activate_train = []
        index_lst = []
        self.u = u
        self.k_bins = k_bins
        self.threshold = threshold
        self.test_score = []

        self.time_limit = time_limit

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.neuron_activate_train.append(temp.copy())
        self.neuron_activate_train = np.concatenate(self.neuron_activate_train, axis=1)
        self.train_label = np.array(label)
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def find_closest_at(self, at, train_ats):
        dist = np.linalg.norm(at - train_ats, axis=1)
        return (min(dist), train_ats[np.argmin(dist)])

    def fit(self, test, label):
        s = time.time()
        start = time.time()
        self.neuron_activate_test = []
        self.test_score = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test = np.concatenate(self.neuron_activate_test, axis=1)

        class_matrix = {}
        all_idx = []
        for i, lb in enumerate(self.train_label):
            if lb not in class_matrix:
                class_matrix[lb] = []
            class_matrix[lb].append(i)
            all_idx.append(i)

        for test_sample, label_sample in tqdm(zip(self.neuron_activate_test, label)):
            end = time.time()
            if end - start >= self.time_limit:
                print("=======================time limit=======================")
                return None
            x = self.neuron_activate_train[class_matrix[label_sample]]
            a_dist, a_dot = self.find_closest_at(test_sample, x)
            y = self.neuron_activate_train[list(set(all_idx) - set(class_matrix[label_sample]))]
            b_dist, _ = self.find_closest_at(
                a_dot, y
            )
            self.test_score.append(a_dist / b_dist)

        e = time.time()
        self.t_collection += e - s
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        x = np.unique(np.digitize(self.test_score, bins))
        rate = len(np.unique(x)) / float(self.k_bins)
        return rate

    def get_sore(self):
        return self.test_score

    def get_u(self):
        return self.u

    def rank_2(self):
        s = time.time()
        res = np.argsort(self.get_sore())[::-1]
        e = time.time()
        self.t_ctm = e - s
        return res

    def rank_fast(self):
        s = time.time()
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        score_bin = np.digitize(self.test_score, bins)
        score_bin_uni = np.unique(score_bin)
        res_idx_arr = []
        for x in score_bin_uni:
            np.random.seed(41)
            idx_arr = np.argwhere(score_bin == x).flatten()
            idx = np.random.choice(idx_arr)
            res_idx_arr.append(idx)
        e = time.time()
        self.t_cam = e - s
        return res_idx_arr


if __name__ == '__main__':
    pass