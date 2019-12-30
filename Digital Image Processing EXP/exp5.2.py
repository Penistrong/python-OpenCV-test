# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 13:55:59
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 14:06:12
# 实验5.2 空域图像锐化
# 使用Sobel和Laplace算子

import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
img = cv.imread('./exp.list/moon.jpg', 0)
# cv.CV_64F 输出图像的深度（数据类型），可以使用-1, 与原图像保持一致 np.uint8
laplacian = cv.Laplacian(img, cv.CV_64F)
# 参数 1,0 为只在 x 方向求一阶导数，最大可以求 2 阶导数。
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# 参数 0,1 为只在 y 方向求一阶导数，最大可以求 2 阶导数。
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# 显示原图减去Laplacian处理后的图像
plt.subplot(2, 3, 3), plt.imshow(np.subtract(img, laplacian), cmap='gray')
plt.title('Original - laplacian'), plt.xticks([]), plt.yticks([])
plt.show()
