#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2019/9/20 10:17
# @Author : hushaobao
# @File   : tools.py

import numpy as np
import matplotlib.pyplot as plt


def normalization(input):
    res = (input - np.min(input)) / (np.max(input) - np.min(input))
    return res


def drawArrow1(A, B):
    '''
    Draws arrow on specified axis from (x, y) to (x + dx, y + dy). 
    Uses FancyArrow patch to construct the arrow.

    The resulting arrow is affected by the axes aspect ratio and limits. 
    This may produce an arrow whose head is not square with its stem. 
    To create an arrow whose head is square with its stem, use annotate() for example:
    Example:
        ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->"))
    '''
    fig = plt.figure()
    ax = fig.add_subplot(121)
    # fc: filling color
    # ec: edge color
    ax.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1],
             length_includes_head=True,  # 增加的长度包含箭头部分
             head_width=0.25, head_length=0.1, fc='r', ec='b')
    # 注意： 默认显示范围[0,1][0,1],需要单独设置图形范围，以便显示箭头
    ax.set_xlim(0,5)
    ax.set_ylim(0,5)
    ax.grid()
    ax.set_aspect('equal')  # x轴y轴等比例
    # Example:
    ax = fig.add_subplot(122)
    ax.annotate("", xy=(B[0], B[1]), xytext=(A[0], A[1]),arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.grid()
    ax.set_aspect('equal')  # x轴y轴等比例
    plt.show()
    plt.tight_layout()
    # 保存图片，通过pad_inches控制多余边缘空白
    plt.savefig('arrow.png', transparent = True, bbox_inches = 'tight', pad_inches = 0.25)

# a = np.array([1, 2])
# b = np.array([3, 4])
# drawArrow1(a, b)
