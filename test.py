#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2019/9/20 9:25
# @Author : hushaobao
# @File   : test.py


import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random


def preProcessDataForOffLineInfer(fileName):
    #导入测试数据
    with open(fileName, "r") as rf:
        file_data = pd.read_csv(rf)
        file_data = np.array(file_data.get_values(), dtype=np.float32)
        print("fileName:", fileName, "  shape of file data:", file_data.shape)
   

    test_xs = []
    for i in range(len(file_data)):
        org_data = file_data[i]

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
        tmp_test_xs = [feature_dis_2d / 100, feature_RS_Power, feature_dis_3d / 100, feature_delta_h,
                        feature_inter_angle / 10, feature_PL / 100, feature_cci, feature_ci, feature_bh / 10]
        test_xs.append(tmp_test_xs)
    #将test_xs转换为numpy数组类型
    test_xs = np.array(test_xs)  
    print("shape of test_xs:", test_xs.shape) 
    return test_xs


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

def test(fileName):
    test_xs = preProcessDataForOffLineInfer(fileName)
    
    #从保存的模型文件中将模型加载回来
    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess, ["serve"], "./model_pb")
      graph = tf.get_default_graph()
      x = sess.graph.get_tensor_by_name('input_x:0')
      y = sess.graph.get_tensor_by_name('myOutput:0')
      infer_y_value = sess.run(y, feed_dict={x: test_xs})
      print("shape of test_y_value:", infer_y_value.shape)
      #保存结果为csv文件      
      np.savetxt('result/' + fileName.split('_')[-1]+"_test_res.csv", infer_y_value, delimiter=',')
      


test(os.path.join("DataSets/test/","test_112501.csv"))
