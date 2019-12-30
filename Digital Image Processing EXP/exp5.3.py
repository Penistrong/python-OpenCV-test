# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 14:07:31
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 19:18:55
# 实验5.3 图像的傅里叶变换
# 使用numpy的fft
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./exp.list/messigray.png', 0)
# 使用np.fft.fft2()将灰度图像变换为复数数组，可选第二个参数，决定切割或扩充(补0等操作)
f = np.fft.fft2(img)
# 上述操作完成后，频率为0的直流分量在f的左上角，为了将其输出在图像中心，需要将它沿2个方向平移N/2
# 这里直接使用fftshift()帮助我们操作，以完成频率变换
fshift = np.fft.fftshift(f)
# 构建振幅
# 取绝对值：将复数变化成实数
# 取对数的目的为了将数据变化到较小的范围（比如0-255）
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# 频域操作 使用一个60X60的矩形窗口对图像进行掩模操作去除低频分量
# 使用 ifftshift() 进行逆平移操作，将直流分量复归左上角，再使用 ifft2() 进行FFT逆变换
rows, cols = img.shape
# 使用双斜杠除号取整，防止单斜杠的浮点类型出现
crow, ccol = rows // 2, cols // 2
fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
# 取绝对值
img_back = np.abs(img_back)
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_back, cmap='gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.show()
