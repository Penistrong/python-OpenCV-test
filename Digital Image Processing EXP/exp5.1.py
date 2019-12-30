# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 13:11:42
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 13:54:35
# 实验5 滤波处理
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 5.1 对图像进行卷积操作，利用cv.filter()
# 采用二维卷积对2D图像实施平滑与锐化操作，采用一个5X5的平均滤波器K
kernel = np.ones((5, 5), np.float32) / 25
img = cv.imread('./exp.list/lena.bmp')
# 采用滤波器K对原图进行处理
dst = cv.filter2D(img, -1, kernel)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
# cv.filter2D(src, dst, kernel, anchor=(-1, -1))
# ddepth –desired depth of the destination image;
# if it is negative, it will be the same as src.depth();
# the following combinations of src.depth() and ddepth are supported:
# src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
# src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
# src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
# src.depth() = CV_64F, ddepth = -1/CV_64F
# when ddepth=-1, the output image will have the same depth as the source.

# 直接使用openCV的blur函数进行均值滤波，但要指定卷积窗口(滤波器)尺寸
blur3 = cv.blur(img, (3, 3))
blur5 = cv.blur(img, (5, 5))
blur7 = cv.blur(img, (7, 7))
plt.subplot(221), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(blur3), plt.title('Blurred 3*3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(blur5), plt.title('Blurred 5*5')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(blur7), plt.title('Blurred 7*7')
plt.xticks([]), plt.yticks([])
plt.show()

# 中值滤波
median3 = cv.medianBlur(img, 3)
median5 = cv.medianBlur(img, 5)
median7 = cv.medianBlur(img, 7)
plt.subplot(221), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(blur3), plt.title('median 3*3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(blur5), plt.title('median 5*5')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(blur7), plt.title('median 7*7')
plt.xticks([]), plt.yticks([])
plt.show()

# 高斯模糊
Gauss = cv.GaussianBlur(img, (5, 5), 0)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(Gauss), plt.title('GaussianBlur')
plt.xticks([]), plt.yticks([])
plt.show()
