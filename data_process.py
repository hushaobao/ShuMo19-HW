#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2019/9/19 16:14
# @Author : hushaobao
# @File   : data_process.py

import os
# import xmls
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from scipy.stats import spearmanr


def computer_Coverage_distance(h_b, Theta_ED, Theta_MD, cell_altitude, altitude):
    pass


def csv_merge(root_dir, csv_list, csv_save_name):
    """
    函数功能：将多个csv文件合并成一个csv文件
    :param root_dir: 训练文件存放文件夹目录
    :param csv_list: csv 文件列索引
    :param csv_save_name: 输出csv文件名
    :return: None
    """

    file_list = os.listdir(root_dir)

    df1 = pd.DataFrame(csv_list).T
    df1.to_csv(csv_save_name, encoding='gbk', header=False, index=False)

    for i in file_list:
        path = root_dir + i
        data = pd.read_csv(path)
        aa = data.iloc[1:, :]  # 跳过标题行
        aa.to_csv(csv_save_name, mode='a', encoding='gbk', header=False, index=False)


def computer_pl(f, h_b, h_r, Clutter_Index, d):
    """
    通过Cost 231 Hata 经验公式计算PL
    :param f: 载波频率
    :param h_b: 基站天线有效高度
    :param alpha: 用户天线高度纠正项
    :param h_ue: 用户天线有效高度
    :param d: 链路距离
    :param c_m: 场景纠正常数
    :return: pl 传播路径损耗
    """
    Clutter_Index = np.array(Clutter_Index)
    f = np.array(f)
    h_b = np.array(h_b) + np.ones(h_r.shape)
    d = np.squeeze(np.array(d)) + 0.0001
    h_r = np.array(h_r) + np.ones(h_r.shape)
    urban_id_list = [10, 11, 12, 13, 14, 16, 18, 20]
    flag = []
    for i in Clutter_Index:
        if i in urban_id_list:
            flag.append(1)
        else:
            flag.append(0)
    c_m = 3
    alpha = (1.1*np.log10(f) - 0.7)*h_r - 1.56*np.log10(f)+0.8
    tmp_1 = 46.3 + 33.9*np.log10(f) - 13.82*np.log10(h_b) - alpha + (44.9-6.55*np.log10(h_b))*np.log10(d)
    tmp_2 = 46.3 + 33.9*np.log10(f) - 13.82*np.log10(h_b) - alpha + (44.9-6.55*np.log10(h_b))*np.log10(d) + c_m
    pl = np.where(flag, tmp_2, tmp_1)
    PL = pd.DataFrame({'PL': pl})
    return PL


def compute_Relative_height(Theta_ED, Theta_MD, distance, h_b, cell_altitude, altitude):
    """
    求取用户所在区域距离天线的相对高度
    :param Theta_ED: 
    :param Theta_MD: 
    :param distance: 基站与用户间的二维距离
    :param h_b: 基站高度
    :param cell_altitude: 基站海拔
    :param altitude: 用户所在区域海拔
    :return: 用户所在区域距离天线的相对高度
    """
    # Range of Theta: [0-38]
    Theta = Theta_ED + Theta_MD
    Theta = np.array(Theta)
    distance = np.squeeze(np.array(distance))
    h_b = np.array(h_b)
    cell_altitude = np.array(cell_altitude)
    altitude = np.array(altitude)
    # embed()
    h_r = h_b - distance*np.tan((Theta/180)*np.pi) + (cell_altitude - altitude)
    delta_h = pd.DataFrame({'delta_h': h_r})
    return delta_h


def computer_distance(cell_x, cell_y, x, y):
    """
    
    :param cell_x， cell_y: 基站所在区域坐标
    :param x， y: 用户所在区域坐标
    :return: 用户与基站之间的二维距离
    """
    cell_x = np.array(cell_x)
    cell_y = np.array(cell_y)
    x = np.array(x)
    y = np.array(y)
    # embed()
    distance = np.sqrt((cell_x - x) ** 2 + (cell_y - y) ** 2)
    dis = pd.DataFrame({'distance': distance})
    return dis


def inter_angle(angle, cell_x, cell_y, x, y):
    """
    
    :param angle: 基站发射线偏角
    :param cell_x， cell_y: 基站所在区域坐标
    :param x， y: 用户所在区域坐标
    :return: 用户所在区域与基站发射线之间的水平夹角
    """

    angle, cell_x, cell_y, x, y = np.array([angle, cell_x, cell_y, x, y])
    results = np.zeros(len(angle))
    for i in range(len(angle)):
        d_angle = angle[i]
        if d_angle == 0 or d_angle == 360:
            d_x = 0
            d_y = 1
        elif d_angle == 180:
            d_x = 0
            d_y = -1
        elif (d_angle <= 90) and (d_angle > 0):
            d_angle = 90 - d_angle
            d_y = np.tan(d_angle*2*np.pi/360)
            d_x = 1
        elif (d_angle > 90) and (d_angle < 180):
            d_angle = d_angle - 90
            d_y = np.tan(d_angle*2*np.pi/360)*(-1)
            d_x = 1
        elif (d_angle > 180) and (d_angle <= 270):
            d_angle = 90 - (d_angle - 180)
            d_y = np.tan(d_angle*2*np.pi/360)*(-1)
            d_x = -1
        elif (d_angle > 270) and (d_angle < 360):
            d_angle = d_angle - 270
            d_y = np.tan(d_angle*2*np.pi/360)
            d_x = -1
        vector1 = np.array([d_x, d_y])
        vector2 = np.array([x[i]-cell_x[i], y[i]-cell_y[i]])
        if np.sum(vector2) == 0:
            results[i] = 0
        else:
            # 两个向量
            Lx = np.sqrt(vector1.dot(vector1))
            Ly = np.sqrt(vector2.dot(vector2))
            # 相当于勾股定理，求得斜线的长度
            cos_angle = vector1.dot(vector2) / (Lx * Ly)
            if cos_angle >1 or cos_angle < -1:
                cos_angle = round(cos_angle)
            # 求得cos_sita的值再反过来计算，绝对长度乘以cos角度为矢量长度，初中知识。。
            angle_ = np.arccos(cos_angle)
            # if np.isnan(angle_):
            #     pass
            #     # embed()
            angle2 = angle_ * 360 / 2 / np.pi
            # 变为角度
            results[i] = angle2
    return results


def resolve_csv(data):
    """
    
    :param data: 通过pb.read_csv(path)所读取的csv文件
    :return: csv各列数据以及所设计特征的求解
    """
    csv_list = ['Cell Index', 'Cell X', 'Cell Y', 'Height', 'Azimuth', 'Electrical Downtilt',
                'Mechanical Downtilt', 'Frequency Band', 'RS Power', 'Cell Altitude',
                'Cell Building Height', 'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
                'Clutter Index', 'RSRP']

    # obtain param

    Cell_X = data['Cell X']
    Cell_Y = data['Cell Y']
    h_b = data['Height']
    Azimuth = data['Azimuth']
    Theta_ED = data['Electrical Downtilt']
    Theta_MD = data['Mechanical Downtilt']
    # theta = Theta_ED + Theta_MD
    f = data['Frequency Band']
    RS = data['RS Power']
    cell_altitude = data['Cell Altitude']
    Cell_Building_Height = data['Cell Building Height']
    Cell_Clutter_id = data['Cell Clutter Index']
    X = data['X']
    Y = data['Y']
    altitude = data['Altitude']
    h_r = data['Building Height']
    Clutter_Index = data['Clutter Index']
    RSRP = data['RSRP']
    # 直线距离
    dis = computer_distance(Cell_X, Cell_Y, X, Y)
    PL = computer_pl(f, h_b, h_r, Clutter_Index, dis)
    # 相对高度
    delta_h = compute_Relative_height(Theta_ED, Theta_MD, dis, h_b, cell_altitude, altitude)
    embed()
    return h_b, f, RS, Cell_Building_Height, Cell_Clutter_id, h_r, Clutter_Index, dis, PL, delta_h, RSRP


csv_save_name = 'result.csv'

root_dir = 'datasets/train_set/train_set/'

csv_list = ['Cell Index', 'Cell X', 'Cell Y', 'Height', 'Azimuth', 'Electrical Downtilt',
            'Mechanical Downtilt', 'Frequency Band', 'RS Power', 'Cell Altitude',
            'Cell Building Height', 'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
            'Clutter Index', 'RSRP']

file_list = os.listdir(root_dir)

data = pd.read_csv(csv_save_name)
resolve_csv(data)
theta = data['Electrical Downtilt'] + data['Mechanical Downtilt']
# obtain param

Cell_X = data[csv_list[1]]
Cell_Y = data[csv_list[2]]
Theta_ED = data[csv_list[5]]
Theta_MD = data[csv_list[6]]
RS = data[csv_list[8]]
h_b = data[csv_list[3]]
h_r = data[csv_list[15]]
cell_altitude = data[csv_list[9]]
altitude = data[csv_list[14]]
X = data[csv_list[12]]
Y = data[csv_list[13]]
Clutter_Index = data[csv_list[16]]
f = data[csv_list[7]]
d = data[csv_list[7]]
RSRP = data[csv_list[17]]
Azimuth = np.array(data[csv_list[4]])
"""

数据分析：

通过IPython的embed()函数在线进行分析

"""




# print(len(Cell_X), len(Cell_Y))

# print(set(Azimuth))

# print(len(X), len(Y))
# embed()
# x = set(Cell_X)
# y = set(Cell_Y)
# print(len(x), len(y))

# plt.scatter(X, Y, c='r', linewidths=0.5)
# plt.scatter(Cell_X, Cell_Y, c='b', linewidths=0.5)
# plt.show()
# cost 231-hata

# 直线距离
dis = computer_distance(Cell_X, Cell_Y, X, Y)
PL = computer_pl(f, h_b, h_r, Clutter_Index, dis)
# 相对高度
delta_h = compute_Relative_height(Theta_ED, Theta_MD, dis, h_b, cell_altitude, altitude)
print(spearmanr(np.squeeze(np.array(PL)), np.array(RSRP)))
embed()
pass
# count = 0

# ----------
# for i in theta:
#     if i >= 90:
#         count += 1
#         print('error')
# print(len(theta))
# print(count)
# --------
# for i in file_list:
#     data = pd.read_csv(root_dir+i)
#     Cell_X = data[csv_list[1]]
#     Cell_Y = data[csv_list[2]]
#     X = data[csv_list[12]]
#     Y = data[csv_list[13]]
#     plt.scatter(Cell_X, Cell_Y, c='r', linewidths=50)
#     plt.scatter(X, Y)
#     plt.show()
#
#     for idx in csv_list:
#         data_tmp = np.array(data[idx])
#         length = len(data_tmp)
#         x = range(length)
#         plt.scatter(x, data_tmp)
#         # plt.plot(data_tmp)
#         plt.show()
#         pass
from scipy.stats import spearmanr