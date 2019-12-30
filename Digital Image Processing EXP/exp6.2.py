# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 20:57:48
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 21:08:34
# 实验6.2 阈值处理
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./exp.list/gradient.jpg', 0)
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# 自适应阈值处理
# 根据图像的每一个小区域计算其对应阈值，再进行处理
# cv.ADAPTIVE_THRESH_MEAN_C 相邻区域平均值
# cv.ADAPTIVE_THRESH_GAUSSIAN_C 相邻区域加权和, 权重为一个Gauss窗口
# BLOCK SIZE 邻域大小
# C 常数，阈值等于计算阈值方法输出后减去该常数
img = cv.imread('./exp.list/dave.jpg', 0)
img = cv.medianBlur(img, 5)

ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY, 11, 2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
