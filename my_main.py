import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from gan.generator_data import generate_one
from gen_data.CifarDau import CifarDau
from gen_data.FashionDau import FashionDau
from gen_data.GanDau import GanDau
from gen_data.MnistDau import MnistDau
from gen_data.SvhnDau import SvhnDau
from my_cov.CovLayers import AnalyzeCustomLayers
from my_cov.CovRanker import CovRanker
from my_cov.MNUBC import MNUBC
from my_cov.cov_analyze import BaseAnalyzer
from my_cov.cov_config import CovConfig
from nc_coverage import metrics
from utils import model_conf
from utils.fid import calculate_fid_given_paths_own
from utils.utils_train import num_to_str, color_print
import tensorflow as tf

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def load_profile(data_name, model_name, target_label):
    model_profile_path = CovConfig.get_model_profile_path(data_name, model_name, "mean{}".format(target_label))
    profile_dict1 = pickle.load(open(model_profile_path, 'rb'))

    model_profile_path = CovConfig.get_model_profile_path(data_name, model_name, "upper_bound{}".format(target_label))
    profile_dict2 = pickle.load(open(model_profile_path, 'rb'))

    model_profile_path = CovConfig.get_model_profile_path(data_name, model_name, "lower_bound{}".format(target_label))
    profile_dict3 = pickle.load(open(model_profile_path, 'rb'))
    return profile_dict1, profile_dict2, profile_dict3


def analyze_all():
    ...
    # # 4. 分析全部
    # # analyzer.analyze_all(data_name, x_right)
    # # analyzer.clear()
    # # mean_pro, max_pro, min_pro = load_profile(data_name, model_name, "_{}".format("all"))
    # # for layer_info, neuron_arr in mean_pro.items():
    # #     print(layer_info)
    # #     show_feature_map(np.array(neuron_arr), title="{}_{}".format("all", layer_info), is_plot=True)
    #


# # 合并优先级序列
# def merge_and_sort_priority_index(priority_dict, wrong_idx_arr):
#     error_num_dict = {}
#     sort_list = []
#     total_error_num = 0
#     for k, v_arr in priority_dict.items():
#         error_num = len((set([item[0] for item in v_arr[:100]]) & set(wrong_idx_arr)))
#         total_error_num += error_num
#         # error_ratio = error_num / len(wrong_idx_arr)
#         error_num_dict[k] = error_num
#         sort_list += v_arr
#     print("1/10 label error_ratio", error_num_dict)
#     print("found total_error_num: {}/{}".format(total_error_num, len(wrong_idx_arr)))
#     sorted_res_list = sorted(sort_list, key=lambda item: item[1], reverse=True)  #
#     print(sorted_res_list[:100])
#     return sorted_res_list


def integrate(x, y):
    area = np.trapz(y=y, x=x)
    return area


def show(res_path, method, include_layer, sorted_res_list, wrong_idx_arr, cov_name, suffix, x_test_len, index_map):
    layer_name = "".join([x[0] for x in include_layer])
    fig_path = "{}/{}_{}_{}_{}.png".format(res_path, cov_name, method, layer_name, suffix)

    # 计算平均random曲线
    x_wrong_len = len(wrong_idx_arr)
    inter = x_test_len // x_wrong_len
    ran_bins = np.linspace(inter / 2, x_test_len - inter / 2, x_wrong_len)
    ran_bins = [int(temp_ran) for temp_ran in ran_bins]

    index_wrong_dict = {}
    if index_map is not None:
        for name, ix_arr in index_map.items():
            temp_list = []
            start = 0
            for ix in ix_arr:
                if ix in wrong_idx_arr:
                    start += 1
                temp_list.append(start)
            index_wrong_dict[name] = temp_list

    res_arr = []
    theo_arr = []
    ran_arr = []
    start = 0
    theo_start = 0
    ran_start = 0

    # for i in range(len(x_test_len)):
    #     if i < len(sorted_res_list):
    #         k, v = sorted_res_list[i]
    #         if k in wrong_idx_arr:
    #             start += 1
    #     if theo_start < len(wrong_idx_arr):
    #         theo_start += 1
    #     if i == ran_bins[ran_start]:  # 如果到了该加错误的时候
    #         ran_start += 1
    #     ran_arr.append(ran_start)
    #     res_arr.append(start)
    #     theo_arr.append(theo_start)

    effective_length = 0
    for i, (k, v) in enumerate(sorted_res_list):
        if k in wrong_idx_arr:
            start += 1
        if theo_start < len(wrong_idx_arr):
            theo_start += 1
        if ran_start < len(ran_bins) and i == ran_bins[ran_start]:  # 如果到了该加错误的时候
            ran_start += 1
        if v > 0:
            effective_length += 1
        ran_arr.append(ran_start)
        res_arr.append(start)
        theo_arr.append(theo_start)

    # print(x_test_len)
    # if len(res_arr) < x_test_len:
    #     res_arr += [res_arr[-1]] * (x_test_len - len(res_arr))
    #     print(len(res_arr))
    # if len(theo_arr) < x_test_len:
    #     theo_arr += [theo_arr[-1]] * (x_test_len - len(theo_arr))
    #     print(len(theo_arr))
    # if len(ran_arr) < x_test_len:
    #     ran_arr += [ran_arr[-1]] * (x_test_len - len(ran_arr))
    #     print(len(ran_arr))
    racu0 = integrate(range(len(theo_arr)), theo_arr)
    racu1 = num_to_str(integrate(range(len(res_arr)), res_arr) / racu0, 2)
    racu3 = num_to_str(integrate(range(len(ran_arr)), ran_arr) / racu0, 2)
    plt.plot(range(len(res_arr)), res_arr, color="red", label="MNAC_{}".format(racu1))
    plt.plot(range(len(theo_arr)), theo_arr, color="gold", label="Theo")
    plt.plot(range(len(ran_arr)), ran_arr, color="black", label="Ran_{}".format(racu3))
    for name, value in index_wrong_dict.items():
        racu = num_to_str(integrate(range(len(value)), value) / racu0, 2)
        plt.plot(range(len(value)), value, label="{}_{}".format(name, racu), alpha=0.5)
    plt.axvline(effective_length, label="effective_length", color='black', linestyle="--")
    plt.title("{}_{}_{}_{}".format(cov_name, method, layer_name, suffix))
    plt.legend()
    plt.savefig(fig_path)
    color_print("fig save path {}".format(fig_path), "blue")
    plt.show()


def mkdir(path):
    '''
    创建指定的文件夹
    :param path: 文件夹路径，字符串格式
    :return: True(新建成功) or False(文件夹已存在，新建失败)
    '''
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' Success!')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' Already exist!')
        return False


def get_cov_index_map(analyze_layers):
    index_map = {}
    params = {
        "data_name": data_name,
        "model_name": model_name
    }
    input_layer = analyze_layers.get_model_input()
    layers, _, _ = analyze_layers.load_compute_layer()
    layers = list(zip(len(layers) * ['conv'], layers))  # 格式要求

    nac = metrics.nac(x_test, input_layer, layers, t=0)
    index_map["nac1_ctm_0"] = nac.rank_2(x_test)
    nac = metrics.nac(x_test, input_layer, layers, t=0.75)
    index_map["nac1_ctm_0.75"] = nac.rank_2(x_test)

    # cov_initer = CovInit(x_train, y_train, params)
    # input_layer = cov_initer.get_input_layer()
    # layers = cov_initer.get_layers()
    # nac = metrics.nac(x_test, input_layer, layers, t=0)
    # index_map["nac2_ctm_0"] = nac.rank_2(x_test)
    # nac = metrics.nac(x_test, input_layer, layers, t=0.75)
    # index_map["nac2_ctm_0.75"] = nac.rank_2(x_test)

    return index_map


# 该函数的目的是生成一个index_map包含不同覆盖条件下的覆盖索引值的字典
#
# 初始化一个空字典
# 初始化一个空字典index_map。
#
# params使用键"data_name"和创建参数字典"model_name"。
#
# input_layer使用对象get_model_input()的方法获取输入层analyze_layers。
#
# 得到load_compute_layer()的方法analyze_layers对象，并将返回的元组解压为layers, _, 和_。layers代表图层信息。
#
# 压缩每个元素layers列表与列表len(layers) 'conv'字符串，创建一个新列表layers。这
#
# nac使用函数计算值metrics.nac，其中x_test是输入数据，input_layer是输入层，layers为图层信息，t=0表示覆盖条件
#
# 应用该rank_2()方法nac对象x_test并存储index_map字典下的键"nac1_ctm_0"。
#
# nac使用计算值metrics.nac函数，其中x_test是输入数据，input_layer是输入层，layers为图层信息，t=0.75表示
#
# 应用rank_2()nac对象的方法x_test并存储index_map字典下的键"nac1_ctm_0.75"。
#
# 创建一个CovInit对象cov_initer并初始化它x_train、y_train、 以及params作为
#
# 得到input_layer使用get_input_layer()对象的方法cov_initer。
#
# 获取图层信息layers使用get_layers()的方法cov_initer目的。
#
# nac使用以下公式计算该值metrics.nac函数，其中x_test是输入数据，input_layer是输入层，layers是层信息，t=0表示覆盖条件
#
# 应用该rank_2()方法nac反对x_test并将结果存储在index_map字典下的键"nac2_ctm_0"。
#
# nac使用以下公式计算该值metrics.nac函数，x_test其中input_layer是输入层，layers是t=0.75表示覆盖条件为0
#
# 应用该rank_2()方法nac反对x_test并将结果存储在index_map字典下的键"nac2_ctm_0.75"。
#
# 返回index_map字典。

def get_dau(data_name):
    if data_name == model_conf.mnist:
        return MnistDau()
    if data_name == model_conf.fashion:
        return FashionDau()
    if data_name == model_conf.svhn:
        return SvhnDau()
    if data_name == model_conf.cifar10:
        return CifarDau()
    if data_name == model_conf.gandata:
        return GanDau()


def load_ori_model():
    model_path = model_conf.get_model_path('mnist', 'LeNet5')
    return load_model(model_path)


def get_gini(model, x_s, ):
    pred_test_prob = model.predict(x_s)
    result = np.argmax(pred_test_prob, axis=1)
    metrics = np.sum(pred_test_prob ** 2, axis=1)
    return 1 - metrics, result


#
# 1. CAM 的实现
# 2. FC 的实现
# 3. 目标检测
# 4. 回归任务
# 5. Gan 随机生成训练集大小相同的张数,,然后看平均, 然后再采样更新覆盖率

# 检测GAn生成数据质量
# 1. 有了一个GAN
# 2. 1000次 1000个数据 训练集 ---> 覆盖率
# 3. for 循环1000次 每次判断 覆盖率有没有增加, 收集这些数据 cam/ctm
# GAN 生成数据是随机的,
# 条件GAN
if __name__ == '__main__':
    is_analyze = False
    is_exp_cov = False
    is_exp_mcov = False
    is_exp_ori_order = False
    th = 0
    # th = -0.5

    # config
    # data_name = model_conf.cifar10
    # data_name = model_conf.fashion
    data_name = model_conf.mnist
    # model_name = model_conf.resNet20
    # model_name = model_conf.LeNet1
    # model_name = model_conf.dcggan
    model_name = model_conf.cDCGAN
    nb_classes = model_conf.fig_nb_classes
    # data
    # dau = get_dau(data_name)
    dau = get_dau('gandata')
    # (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
    x_train, x_test = dau.load_data(use_norm=True)
    # model
    # print(model_conf.get_model_path(data_name, model_name).encode('utf-8'))
    # 加载模型
    model = load_model(model_conf.get_model_path(data_name, model_name))
    # print(model.evaluate(x_test, keras.utils.np_utils.to_categorical(y_test, 10)))

    print(model.summary())
    base_output_dir = "_res/{}/{}".format(data_name, model_name)

    # prob_matrixc = model.predict(x_test)
    # ys_psedu = np.argmax(prob_matrixc, axis=1)  #
    # wrong_idx_arr = np.array(range(0, len(y_test)))[y_test != ys_psedu]
    # print(wrong_ix)
    # analyze
    # 1.只分析预测正确的数据
    # x_right, x_wrong, y_right, y_wrong = split_data_by_right(model, x_train, y_train)
    # 2. 分析
    # analyze_layers = AnalyzeConfigLayers(model, model_name)
    # dcgan
    # include_layer = ['conv2d_1','activation_3','up_sampling2d_2','conv2d_2','activation_4']
    # Cdcgan
    include_layer = ['dense_2', 'batch_normalization_3', 'reshape_1', 'conv2d_transpose_3', 'batch_normalization_4',
                     'leaky_re_lu_6', 'conv2d_transpose_4', 'batch_normalization_5', 'leaky_re_lu_7',
                     'conv2d_transpose_5']

    analyze_layers = AnalyzeCustomLayers(model, model_name, include_layer)
    analyzer = BaseAnalyzer(analyze_layers)
    # if is_analyze:
    #     analyzer.analyze_by_label(data_name, x_right, y_right, split_num=10)
    #     analyzer.clear()
    noise = np.load("./data/{}/{}_noise_data.npy".format(data_name, model_name))
    image = np.load("./data/{}/{}_images_data.npy".format(data_name, model_name))
    if is_analyze:
        analyzer.analyze_all(data_name, noise)
        analyzer.clear()
    # 3. 查看结果
    # for i in range(10):
    #     mean_pro, max_pro, min_pro = load_profile(data_name, model_name, "_{}".format(i))
    #     for layer_info, neuron_arr in mean_pro.items():
    #         print(layer_info)
    #         show_feature_map(np.array(neuron_arr), title="{}_{}".format(i, layer_info), is_plot=True)
    #
    # 3.新版查看结果
    mean_pro, max_pro, min_pro = load_profile(data_name, model_name, "_{}".format('all'))

    # for layer_info, neuron_arr in mean_pro.items():
    #     print(layer_info)
    #     show_feature_map(np.array(neuron_arr), title="_{}".format(layer_info), is_plot=True)
    # 实验
    index_map = None
    # exp_layers = AnalyzeCustomLayers(model, model_name, include_layer=["act"])
    # analyze_layers = exp_layers

    if is_exp_cov:
        index_map = get_cov_index_map(analyze_layers)
    if is_exp_ori_order:
        index_map["ori_order"] = list(range(len(x_test)))

    # # mean_pro, max_pro, min_pro = load_profile(data_name, model_name, "_{}".format(target_label))
    # cov = MNAC(th)  # 创建指标
    cov = MNUBC()  # 创建指标
    # cov = MNLBC()  # 创建指标
    # method = "cam"
    method = "ctm"

    cov_ranker = CovRanker(cov, output_dir=base_output_dir)  # 创建排序器,必须得cov初始化了之后才能排序
    is_by_label = False
    if is_exp_mcov:
        cov.init(data_name, analyze_layers, base_output_dir, nb_classes, is_by_label)  # 与模型关联 加载profile
        cov.fit()  # 计算覆盖率
        # # # 可视化
        # for label, layer_dict in mnac.neuron_coverage_dict.items():
        #     for layer_info, neuron_matrix_arr in layer_dict.items():  # 每一层
        #         title = "{}_{}".format(label, layer_info)
        #         show_feature_map(np.array(neuron_matrix_arr), title=title, is_plot=False, is_save=True,
        #                          save_path="fig/{}.png".format(title))
        # 输出分析信息
        # 测试
        # 输出测试集 每个数据的覆盖率大小
        # neuron_test_cov_num_dict = cov_ranker.test_by_label(x_test, y_test, nb_classes, batch_num=100,
        #                                                     is_save_matrix=True)
        # one_noise = noise[901].reshape(1,100)

        neuron_test_cov_num_dict = cov_ranker.test_all(noise, batch_num=1, is_save_matrix=True)

        print(neuron_test_cov_num_dict)
        # ctm排序
        # cam 排序
        if method == "cam":
            priority_dict = cov_ranker.cam_all(neuron_test_cov_num_dict, is_save=True)  # 所有的数据一起排序
        else:
            priority_dict = cov_ranker.ctm_all(neuron_test_cov_num_dict, is_save=True)  # 所有的数据一起排序

        # priority_dict = cov.test_by_label(x_test, y_test, nb_classes, batch_num=100, is_save=True)  # , batch_num=10
    else:
        1
        # priority_dict = cov_ranker.read_priority_index()  # 加载优先级序列 #TODO:
    # print(priority_dict)
    # sorted_res_list = merge_and_sort_priority_index(priority_dict, wrong_idx_arr)  # 合并优先级序列
    # print(len(priority_dict))
    # show_image_num = 10
    # for num in range(show_image_num):
    #     plt.imshow(image[num])
    # res_filename = r'_res\mnist\dcggan\mnlbc\ctm\all\res.txt'
    res_filename = r'_res\mnist\cDCGAN\mnubc\ctm\all\res.txt'
    is_exp = True
    if is_exp:
        index = []
        mcov = []
        length = 0
        with open(res_filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                    pass
                p_tmp, E_tmp = [int(i) for i in lines.split()]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                length = length + 1
                index.append(p_tmp)  # 添加新读取的数据
                mcov.append(E_tmp)
                pass
        pos = np.array(index)  # 将数据从list类型转换为array类型。
        Efield = np.array(mcov)
        pass

        temp = 40
        is_exp_onecov = True
        if is_exp_onecov:
            cov.init(data_name, analyze_layers, base_output_dir, nb_classes, is_by_label)  # 与模型关联 加载profile
            cov.fit()  # 计算覆盖率
        is_exp_train = False
        now_mcov = Efield[temp]
        # 演示代码
        if is_exp_train:
            after_mcov = 0
            noise_tmp = noise[pos[temp]]
            noise_tmp = noise_tmp.reshape(1, 110)
            # while now_mcov>after_mcov:
            add_cov_num_o, neuron_test_cov_num_dict1 = cov_ranker.test_one(noise_tmp, is_save_matrix=True)
            # 计算覆盖率
            image1 = np.stack((image[pos[temp]],) * 3, axis=-1)
            image2 = np.stack((generate_one(model, noise_tmp),) * 3, axis=-1)
            fid = calculate_fid_given_paths_own(image1.reshape(1, 28, 28, 3), image2.reshape(1, 28, 28, 3))
            # print("fid ", fid)
            plt.figure()
            plt.axis('off')
            plt.title("NeuronCoverage: " + str(add_cov_num_o) + "  FID: " + str(fid))
            plt.imshow(generate_one(model, noise_tmp).reshape(28, 28), cmap='gray')
            plt.show()
            add_cov_num = sys.maxsize

            while add_cov_num_o < add_cov_num:
                noise_tmp[0] = noise_tmp[0] + 0.1
                add_cov_num, neuron_test_cov_num_dict = cov_ranker.test_one(noise_tmp, is_save_matrix=True)
                generated_image = generate_one(model, noise_tmp)
                # print(generated_image.shape)
                image2 = np.stack((generated_image,) * 3, axis=-1)
                fid = calculate_fid_given_paths_own(image1.reshape(1, 28, 28, 3), image2.reshape(1, 28, 28, 3))
                add_cov_num_o = add_cov_num
                plt.figure()
                plt.title("NeuronCoverage: " + str(add_cov_num) + "  FID: " + str(fid))
                plt.axis('off')
                plt.imshow(generated_image.reshape(28, 28), cmap='gray')
                plt.show()
        # noise_tmp = noise[pos[temp]]
        # noise_tmp = noise_tmp.reshape(1, 100)
        begin = 45
        end = 46
        is_generate_images = True

        img_path = r'_img\mnist\cDCGAN'

        datasets = 'mnist'

        model_name = 'LeNet5'
        is_random = True
        if is_generate_images:
            for temp in range(begin, end):

                img_path_temp = "{}\_{}".format(img_path, str(temp))
                mkdir(img_path_temp)
                noise_tmp = noise[pos[temp]]
                noise_tmp = noise_tmp.reshape(1, 110)
                # 加载letnet5
                print(noise_tmp)
                condition = np.argmax(noise_tmp[0][100:])
                print('condition:')
                print(condition)
                model_letNet5 = load_ori_model()
                gini, res = get_gini(model_letNet5, image[pos[temp]].reshape(1, 28, 28, 1))
                gini = round(gini[0] * 1000)
                res = res[0]

                # print("%.3f"%gini[0])
                # print(res)
                after_mcov = 0
                num = 0
                turns = 0
                add_cov_num_o, neuron_test_cov_num_dict1 = cov_ranker.test_one(noise_tmp, is_save_matrix=True)
                add_cov = 0
                fig_path = "{}\{}_{}_{}_{}_{}_{}_{}.png".format(img_path_temp, str(num), str(turns), str(add_cov_num_o),
                                                                "-1", str(add_cov), str(gini), str(res))

                plt.figure()
                plt.axis('off')
                plt.title("NeuronCoverage: " + str(add_cov_num_o))
                plt.imshow(image[pos[temp]].reshape(28, 28), cmap='gray')
                plt.savefig(fig_path)

                IMG_NUM_MAX = 2000
                add_cov_num = sys.maxsize
                dimension_num = 0
                prev_cov_num = add_cov_num_o
                dimensions = np.arange(0, 100, 1)
                if is_random:
                    np.random.choice(dimensions, 100)
                image1 = np.stack((image[pos[temp]],) * 3, axis=-1).reshape(1, 28, 28, 3)
                while num < IMG_NUM_MAX:
                    if dimension_num >= 100:
                        break
                    print("dimension")
                    print(dimensions[dimension_num])
                    num += 1
                    noise_tmp[0][dimensions[dimension_num]] = noise_tmp[0][dimensions[dimension_num]] + 0.001
                    add_cov_num, neuron_test_cov_num_dict = cov_ranker.test_one(noise_tmp, is_save_matrix=True)
                    # a = neuron_test_cov_num_dict & prev_neuron_test_cov_num_dict
                    # np.sum(a) > np.sum(prev_neuron_test_cov_num_dict)
                    # prev_neuron_test_cov_num_dict = a
                    print("NeuronCoverage")
                    print(add_cov_num)

                    # if add_cov_num <= prev_cov_num:
                    #     dimension_num += 1
                    #     print("Skip")
                    #     continue
                    generated_image = generate_one(model, noise_tmp)
                    image2 = np.stack((generated_image,) * 3, axis=-1).reshape(1, 28, 28, 3)
                    fid = calculate_fid_given_paths_own(image1, image2)
                    turns += 1
                    gini, res = get_gini(model_letNet5, generated_image.reshape(1, 28, 28, 1))
                    gini = round(gini[0] * 1000)
                    res = res[0]
                    plt.figure()
                    plt.title("NeuronCoverage: " + str(add_cov_num))
                    plt.axis('off')
                    plt.imshow(generated_image.reshape(28, 28), cmap='gray')
                    # si = get_inception_score(list(generated_image))
                    # 编号_迭代轮数_覆盖率_当前纬度_增加量_gini_预测_fid.png
                    fig_path = "{}\{}_{}_{}_{}_{}_{}_{}_{}.png".format(img_path_temp, str(turns), str(num),
                                                                       str(add_cov_num),
                                                                       str(dimensions[dimension_num]),
                                                                       str(add_cov_num - add_cov_num_o), str(gini),
                                                                       str(res), str(fid))
                    plt.savefig(fig_path)
                    prev_cov_num = add_cov_num

    # plt.figure()
    # for i in range(1,9):
    #     plt.subplot(4, 4, i)
    #     plt.title("NC:"+str(mcov[i-1]))
    #     plt.axis('off')
    #     plt.imshow(image[index[i-1]].reshape(28, 28), cmap='gray')
    #     if 17-i==15:
    #         plt.subplot(4, 4, 17 - i)
    #         plt.title("NC:" + str(mcov[-i]))
    #         plt.axis('off')
    #         plt.imshow(image[index[-19]].reshape(28, 28), cmap='gray')
    #         continue
    #     plt.subplot(4, 4, 17-i)
    #     plt.title("NC:" +str(mcov[-i]))
    #     plt.axis('off')
    #     plt.imshow(image[index[-i]].reshape(28, 28), cmap='gray')
    # plt.show()

    # 1.
    # 覆盖引导, 找出高覆盖数据 √
    # 2.
    # 利用现有模型, 判断高覆盖数据中哪些出错了 √
    # 3.
    # 找到出错数据和它对应的输入维度向量, 作为初始输入种子(如果没有错误的, 用高覆盖的也行)
    # 4.
    # 将输入向量沿着一个方向, (或一个维度)
    # 进行一次小步长改变(正负均可), 每次改变观察生成结果, 如果还是错的, 就一直朝着改方向改变
    # 5.
    # 重复4, 直到试过所有的方向(维度, 包括正负), 观察生成样本的情况, 和错误情况
    # plt.show()
    # print(len(x_test))
    # print(len(priority_dict.values()[0]))
    # assert len(priority_dict.values()[0]) == len(x_test)  # mark 先这么设计

    # include_layer = analyze_layers.include_layer
    # show(cov.output_dir, method, include_layer, priority_dict["all"], wrong_idx_arr, cov.coverage_name, th,
    #      len(x_test), index_map)  # 展示结果

# if abs_ix in wrong_ix:
#     flag = True
#     wrong_message = "wrong"
# plt.title("{}_{}".format(temp[1], wrong_message))
# fig_path = "{}/{}_{}_{}.png".format(output_dir, il, abs_ix, wrong_message)
# plt.imshow(x_test[abs_ix])
# plt.savefig(fig_path)
# plt.close()
# if il == 300:
#     break
# print(num, num / len(wrong_idx_arr))
# 1000/10000 0.1 0.24
# 2000/10000 0.2 0.41
# 3000/10000 0.3 0.51
# for x, y in zip(x_test, y_test):
