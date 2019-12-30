# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 14:38:22
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 20:18:44
# 实验5 思考题
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

# 实现不同均值滤波核值下的操作，使用5X5尺寸
img = cv.imread('./exp.list/lena.bmp')
k_1 = np.matrix([[1, 2, 3, 2, 1], [2, 4, 7, 4, 2], [3, 7, 14, 7, 3], [2, 4, 7, 4, 2], [1, 2, 3, 2, 1]]) / 90
dst = cv.filter2D(img, -1, k_1)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging-using k_1')
plt.xticks([]), plt.yticks([])
plt.show()

# 对已进行傅里叶变换并移动了中心的频率域图来说，从中心到边缘，频率上依次是从低频到高频
# 由此要对频率域进行滤波处理只要关心低频和高频分界点的阈值范围即可，用中心模板进行滤波


def fHighPass(img):
    # 频率域高通滤波, 模板样式：一个矩形，中心一圆，为黑色，表示低频至0;其他区域为白色，也就是去除低频
    # 此处使用半径30的尺寸
    rows, cols = img.shape
    # 如果不是灰度图像则转换
    if img.dtype != np.uint8:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.ones(img.shape, np.uint8)
    kernel[rows // 2 - 30:rows // 2 + 30, cols // 2 - 30:cols // 2 + 30] = 0
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # 乘上H(U,V),即此处的kernel
    fshift = fshift * kernel
    ifshift = np.fft.ifftshift(fshift)
    inv_f = np.fft.ifft2(ifshift)
    # 对复数取绝对值以显示
    res_Img = np.abs(inv_f)
    # 调整范围便于显示
    res_Img = (res_Img - np.amin(res_Img)) / (np.amax(res_Img) - np.amin(res_Img))
    return res_Img


def fLowPass(img):
    # 频率域低通滤波, 模板样式：一个矩形，中心一圆，为白色;其他区域为黑色，表示高频置0，也就是去除高频
    # 此处使用半径30的尺寸
    rows, cols = img.shape
    # 如果不是灰度图像则转换
    if img.dtype != np.uint8:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.zeros(img.shape, np.uint8)
    # 中心为1
    kernel[rows // 2 - 30:rows // 2 + 30, cols // 2 - 30:cols // 2 + 30] = 1
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # 乘上H(U,V),即此处的kernel
    fshift = fshift * kernel
    ifshift = np.fft.ifftshift(fshift)
    inv_f = np.fft.ifft2(ifshift)
    # 对复数取绝对值以显示
    res_Img = np.abs(inv_f)
    # 调整范围便于显示
    res_Img = (res_Img - np.amin(res_Img)) / (np.amax(res_Img) - np.amin(res_Img))
    return res_Img


def fBandPass(img):
    # 频率域带通滤波器, 高通低通结合一部分,保留部分低频与高频
    rows, cols = img.shape
    # 如果不是灰度图像则转换
    if img.dtype != np.uint8:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 低通部分
    kernel_L = np.ones(img.shape, np.uint8)
    # 中心为0
    kernel_L[rows // 2 - 30:rows // 2 + 30, cols // 2 - 30:cols // 2 + 30] = 0
    # 高通部分
    kernel_H = np.zeros(img.shape, np.uint8)
    # 中心为1
    kernel_H[rows // 2 - 30:rows // 2 + 30, cols // 2 - 30:cols // 2 + 30] = 1

    kernel = kernel_L * kernel_H

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # 乘上H(U,V),即此处的kernel
    fshift = fshift * kernel
    ifshift = np.fft.ifftshift(fshift)
    inv_f = np.fft.ifft2(ifshift)
    # 对复数取绝对值以显示
    res_Img = np.abs(inv_f)
    # 调整范围便于显示
    res_Img = (res_Img - np.amin(res_Img)) / (np.amax(res_Img) - np.amin(res_Img))
    return res_Img


# 对一幅图像使用高通、带通、低通滤波，分别表示为BGR分量后叠加输出图像
img = cv.imread('./exp.list/child.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 将灰度图输入三个频率域滤波函数
B = fHighPass(img_gray)
G = fBandPass(img_gray)
R = fLowPass(img_gray)
res = cv.merge([B, G, R])

plt.subplot(131), plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_gray, cmap='gray')
plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(res)
plt.title('[HP, BP, LP] Image'), plt.xticks([]), plt.yticks([])
plt.show()
