# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-28 11:36:09
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-28 12:20:51

import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('./exp.list/Fig6A.jpg')
img2 = cv2.imread('./exp.list/Fig6B.jpg')

img_hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)     # bgr转hsv
img_hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

color = ('h', 's', 'v')

# 处理每一个分量
for i, col in enumerate(color):
    # 计算两张图的同一分量下的累积直方图
    # histr = cv2.calcHist([img_hsv1], [i], None, [256], [0, 256])
    hist1, bins = np.histogram(img_hsv1[:, :, i].ravel(), 256, [0, 256])
    hist2, bins = np.histogram(img_hsv2[:, :, i].ravel(), 256, [0, 256])
    cdf1 = hist1.cumsum()  # 灰度值0-255的累计值数组
    cdf2 = hist2.cumsum()
    cdf1_hist = hist1.cumsum() / cdf1.max()  # 灰度值的累计值的比率
    cdf2_hist = hist2.cumsum() / cdf2.max()

    diff_cdf = [[0 for j in range(256)] for k in range(256)]  # diff_cdf 里是每2个灰度值比率间的差值
    for j in range(256):
        for k in range(256):
            diff_cdf[j][k] = abs(cdf1_hist[j] - cdf2_hist[k])

    lut = [0 for j in range(256)]        # 映射表
    for j in range(256):
        min = diff_cdf[j][0]
        index = 0
        for k in range(256):            # 直方图规定化的映射原理
            if min > diff_cdf[j][k]:
                min = diff_cdf[j][k]
                index = k
        lut[j] = ([j, index])

    h = int(img_hsv1.shape[0])
    w = int(img_hsv1.shape[1])
    for j in range(h):                   # 对原图像进行灰度值的映射
        for k in range(w):
            img_hsv1[j, k, i] = lut[img_hsv1[j, k, i]][1]


hsv_img1 = cv2.cvtColor(img_hsv1, cv2.COLOR_HSV2BGR)   # hsv转bgr
hsv_img2 = cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2BGR)

# 注意使用matpotlib的imshow函数，对彩色图像默认是RGB格式，而CV2读取时是以BGR格式
# 使用img=img[...,::-1] 或 库函数 cvtColor()中参数指明为cv2.COLOR_BGR2RGB即可
plt.subplot(231), plt.imshow(img1[..., ::-1])
plt.title('Src Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(img2[..., ::-1])
plt.title('Src Image 2'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(hsv_img1[..., ::-1])
plt.title('Result Image'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.hist(img1.ravel(), 256, [0, 256])
plt.subplot(235), plt.hist(img2.ravel(), 256, [0, 256])
plt.subplot(236), plt.hist(hsv_img1.ravel(), 256, [0, 256])
plt.show()