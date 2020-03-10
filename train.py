# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2019/9/20 9:25
# @Author : hushaobao
# @File   : train.py


import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
from IPython import embed


def computer_pl(f, h_b, h_r, Clutter_Index, d):
    """
    :param f: 载波频率
    :param h_b: 基站天线有效高度
    :param alpha: 用户天线高度纠正项
    :param h_r: 用户天线有效高度
    :param d: 链路距离
    :param c_m: 场景纠正常数
    :return: pl 传播路径损耗
    """

    f = f + 0.0001
    h_b = h_b + 0.0001
    d = d + 0.0001
    h_r = np.array(h_r) + np.ones(h_r.shape)
    urban_id_list = [10, 11, 12, 13, 14, 16, 18, 20]

    c_m = 3
    alpha = (1.1 * np.log10(f) - 0.7) * h_r - 1.56 * np.log10(f) + 0.8
    tmp_1 = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_b) - alpha + (44.9 - 6.55 * np.log10(h_b)) * np.log10(d)
    tmp_2 = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_b) - alpha + (44.9 - 6.55 * np.log10(h_b)) * np.log10(
        d) + c_m
    if Clutter_Index in urban_id_list:
        PL = tmp_2
    else:
        PL = tmp_1
    return PL


def inter_angle(d_angle, cell_x, cell_y, x, y):
    if d_angle == 0 or d_angle == 360:
        d_x = 0
        d_y = 1
    elif d_angle == 180:
        d_x = 0
        d_y = -1
    elif (d_angle <= 90) and (d_angle > 0):
        d_angle = 90 - d_angle
        d_y = np.tan(d_angle * 2 * np.pi / 360)
        d_x = 1
    elif (d_angle > 90) and (d_angle < 180):
        d_angle = d_angle - 90
        d_y = np.tan(d_angle * 2 * np.pi / 360) * (-1)
        d_x = 1
    elif (d_angle > 180) and (d_angle <= 270):
        d_angle = 90 - (d_angle - 180)
        d_y = np.tan(d_angle * 2 * np.pi / 360) * (-1)
        d_x = -1
    elif (d_angle > 270) and (d_angle < 360):
        d_angle = d_angle - 270
        d_y = np.tan(d_angle * 2 * np.pi / 360)
        d_x = -1
    # if d_x == 0 and d_y == 0:
    #     print()
    #     pass
    vector1 = np.array([d_x, d_y])
    vector2 = np.array([x - cell_x, y - cell_y])
    if np.sum(vector2) == 0:
        results = 0
    else:
        # 两个向量
        Lx = np.sqrt(vector1.dot(vector1))
        Ly = np.sqrt(vector2.dot(vector2))

        cos_angle = vector1.dot(vector2) / (Lx * Ly)
        if cos_angle > 1 or cos_angle < -1:
            cos_angle = round(cos_angle)
            # embed()
            # print()
            # pass
        angle_ = np.arccos(cos_angle)
        # if np.isnan(angle_):
        #     pass
        results = angle_ * 360 / 2 / np.pi

    return results


def preProcessDataForTraining():

    # *****************************  read csv file  ********************************************
    # load train DataSets
    Data_path = "DataSets/train/"
    total_train_set_data = []
    for train_set_file_name in os.listdir(Data_path):
        with open(os.path.join('train_set', train_set_file_name), "r") as rf:
            file_data = pd.read_csv(rf)
            file_data = np.array(file_data.get_values(), dtype=np.float32)
            print("fileName:", train_set_file_name, "  shape of file data:", file_data.shape)
            total_train_set_data.extend(file_data)
    total_train_set_data = np.array(total_train_set_data)
    print("shape of total_train_set_data data:", total_train_set_data.shape)
    # Cell Index-0, Cell X-1, Cell Y-2, Height-3, Azimuth-4, Electrical Downtilt-5, Mechanical Downtilt-6,
    # Frequency Band-7, RS Power-8, Cell Altitude-9 , Cell Building Height-10, Cell Clutter Index-11, X-12,
    # Y-13, Altitude-14, Building Height-15, Clutter Index-16, RSRP-17

    # Constructing Data Feature
    total_train_xs = []
    total_train_ys = []
    for i in range(len(total_train_set_data)):
        org_data = total_train_set_data[i]
        # ************************************  Constructing Data Feature  ***************************************
        # 2-D Distance feature
        feature_dis_2d = np.sqrt(np.power(org_data[12] - org_data[1], 2) + np.power(org_data[13] - org_data[2], 2))
        # RS Power feature
        feature_RS_Power = org_data[8]
        # Optional parameters
        # Cell Clutter Index
        feature_cci = org_data[11]
        feature_ci = org_data[16]
        feature_bh = org_data[15]
        # 3-D Distance feature
        feature_dis_3d = np.sqrt(np.power(org_data[12] - org_data[1], 2)
                                 + np.power(org_data[13] - org_data[2], 2)
                                 + np.power(org_data[14] - org_data[9], 2))

        feature_delta_h = org_data[3] - feature_dis_2d * np.tan(((org_data[5] + org_data[6]) / 180) * np.pi) + (
        org_data[9] - org_data[14])

        feature_inter_angle = inter_angle(org_data[4], org_data[12], org_data[13], org_data[1], org_data[2])

        feature_PL = computer_pl(org_data[7], org_data[3], org_data[15], org_data[16], feature_dis_2d)
        tmp_train_xs = [feature_dis_2d / 100, feature_RS_Power, feature_dis_3d / 100, feature_delta_h,
                        feature_inter_angle / 10, feature_PL / 100, feature_cci, feature_ci, feature_bh / 10]
        # 预测真值GT
        tmp_train_ys = [org_data[17]]
        total_train_xs.append(tmp_train_xs)
        total_train_ys.append(tmp_train_ys)
    # 将total_train_xs、total_train_ys转换为numpy数组类型
    total_train_xs = np.array(total_train_xs)
    total_train_ys = np.array(total_train_ys)
    print("shape of total_train_xs:", total_train_xs.shape)
    print("shape of total_train_ys:", total_train_ys.shape)

    return total_train_xs, total_train_ys


def train():

    # ********************  参数初始化  ***********************************

    keep_prob = 0.5
    batch_size = 128
    step = 42000
    lr = 0.005
    input_num = 9  # 因为预处理中只构造了3个特征
    output_num = 1  # 因为要预测的只有1个值

    # 网络输入占位符
    x = tf.placeholder(tf.float32, [None, input_num], name="input")
    # 输出占位符
    y = tf.placeholder(tf.float32, [None, output_num])

    # **************************  构建深度神经网络  **************************

    # 网络输入层

    # 权重、偏置初始化
    w1 = tf.Variable(tf.truncated_normal([input_num, 40], stddev=0.01))
    b1 = tf.Variable(tf.zeros([1, 40]) + 0.01)
    w_b1 = tf.matmul(x, w1) + b1
    # 非线性激活函数relu
    l1 = tf.nn.relu(w_b1)
    # dropout随机失活，防止过拟合
    l1 = tf.nn.dropout(l1, keep_prob)

    # 隐层1

    w2 = tf.Variable(tf.truncated_normal([40, 20], stddev=0.01))
    b2 = tf.Variable(tf.zeros([1, 20]) + 0.1)
    w_b2 = tf.matmul(l1, w2) + b2
    l2 = tf.nn.relu(w_b2)

    l2 = tf.nn.dropout(l2, keep_prob)

    # 隐层
    # w2_1 = tf.Variable(tf.truncated_normal([20, 10], stddev=0.01))
    # b2_1 = tf.Variable(tf.zeros([1, 10]) + 0.1)
    # w_b2_1 = tf.matmul(l2, w2_1) + b2_1
    # l2_1 = tf.nn.relu(w_b2_1)

    # l2_1 = tf.nn.dropout(l2_1, keep_prob)

    # 输出层
    w3 = tf.Variable(tf.truncated_normal([20, output_num], stddev=0.01))
    b3 = tf.Variable(tf.zeros([1, output_num]))

    pre = tf.matmul(l2, w3) + b3
    # 将输出节点 pre 命名为output
    tf.identity(pre, name="output")

    # *******************  构造待优化的损失函数  **************************

    # 采用RMSE
    loss = tf.reduce_mean(tf.sqrt(tf.pow(pre - y, 2)))
    # 选择Adam优化算法最小化cost
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

    # *******************  创建tensor会话，训练网络
    with tf.Session() as sess:
        # 网络参数初始化
        init = tf.initialize_all_variables()
        sess.run(init)
        # 导入预处理之后的训练数据
        total_train_xs, total_train_ys = preProcessDataForTraining()
        for i in range(step):
            # 从预处理的数据集中随机抽取batch_size个样本进行训练

            sample_idxs = random.choices(range(len(total_train_xs)), k=batch_size)
            batch_xs = []
            batch_ys = []
            for idx in sample_idxs:
                batch_xs.append(total_train_xs[idx])
                batch_ys.append(total_train_ys[idx])
            batch_xs = np.array(batch_xs)
            batch_ys = np.array(batch_ys)
            # feed训练数据进训练
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            cost_value = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
            print("after iter:", i, " cost:", cost_value)

        # 保存graph、参数至.pb文件
        tf.saved_model.simple_save(
            sess,
            "./model/",
            inputs={"myInput": x},  # model输入节点x
            outputs={"myOutput": pre}  # model输出节点y
        )


if __name__ == "__main__":
    # 指定GPU 0 运行
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()

