# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-26 20:21:16
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-26 20:59:22

# 统计直方图，使用cv2.calcHist(images[], channels, mask, histSize, range[], hist[], accumulate[])统计一副图像的直方图
# channels： 输入[0]表示灰度图， 输入[[0],[1],[2]]对应彩色图的 BGR

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 4.1.1 绘制灰度直方图
img = cv.imread('./exp.list/Fig6B.jpg', 0)
hist = cv.calcHist([img], [0], None, [256], [0, 256])
hist2, bins = np.histogram(img.ravel(), 256, [0, 256])
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

# 4.1.2绘制三个色彩分量的直方图
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    print(i)
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=color)
    plt.xlim([0, 256])
plt.show()

# 4.1.3绘制累积直方图
img3 = cv.imread('./exp.list/flowers.tif', 0)
# flatten() 一维化
hist, bins = np.histogram(img3.flatten(), 256, [0, 256])
# 计算累积直方图
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(img3.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()
