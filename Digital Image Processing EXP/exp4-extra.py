# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 09:56:35
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 12:46:20
# 对彩色图像进行均衡化 实验4思考题1、2
import cv2 as cv
import numpy as np
import math

np.seterr(invalid='ignore')


def BGR2HSI(rgb_lwpImg):
    # cvtColor没有BGR转HSI，故定义一个
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv.split(rgb_lwpImg)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j] - g[i, j]) + (r[i, j] - b[i, j]))
            den = np.sqrt((r[i, j] - g[i, j])**2 + (r[i, j] - b[i, j]) * (g[i, j] - b[i, j]))
            theta = float(np.arccos(num / den))

            if den == 0:
                H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2 * 3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j] + g[i, j] + r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / sum

            H = H / (2 * 3.14159265)
            I = sum / 3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H * 255
            hsi_lwpImg[i, j, 1] = S * 255
            hsi_lwpImg[i, j, 2] = I * 255
    return hsi_lwpImg


def HSI2BGR(hsi_img):
    # HSI图像转回BGR图像
    h = int(hsi_img.shape[0])
    w = int(hsi_img.shape[1])
    H, S, I = cv.split(hsi_img)
    H = H / 255.0
    S = S / 255.0
    I = I / 255.0
    bgr_img = hsi_img.copy()
    B, G, R = cv.split(bgr_img)
    for i in range(h):
        for j in range(w):
            if S[i, j] < 1e-6:
                R = I[i, j]
                G = I[i, j]
                B = I[i, j]
            else:
                H[i, j] *= 360
                if H[i, j] > 0 and H[i, j] <= 120:
                    B = I[i, j] * (1 - S[i, j])
                    R = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j] * math.pi / 180)) /
                                   math.cos((60 - H[i, j]) * math.pi / 180))
                    G = 3 * I[i, j] - (R + B)
                elif H[i, j] > 120 and H[i, j] <= 240:
                    H[i, j] = H[i, j] - 120
                    R = I[i, j] * (1 - S[i, j])
                    G = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j] * math.pi / 180)) /
                                   math.cos((60 - H[i, j]) * math.pi / 180))
                    B = 3 * I[i, j] - (R + G)
                elif H[i, j] > 240 and H[i, j] <= 360:
                    H[i, j] = H[i, j] - 240
                    G = I[i, j] * (1 - S[i, j])
                    B = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j] * math.pi / 180)) /
                                   math.cos((60 - H[i, j]) * math.pi / 180))
                    R = 3 * I[i, j] - (G + B)
            bgr_img[i, j, 0] = B * 255
            bgr_img[i, j, 1] = G * 255
            bgr_img[i, j, 2] = R * 255
    return bgr_img


# 在HSI空间对亮度分量均衡化, I即Intensity(亮度, 密度)
img = cv.imread('./exp.list/Fig6.png')
hsi_img = BGR2HSI(img)
# cv.imshow('BGR2HSI', hsi_img)
# 取出I分量
(H, S, I) = cv.split(hsi_img)
# 或者直接对矩阵进行操作
# I = hsi_img[:,:,2]
equ_I = cv.equalizeHist(I)
# 叠加处理后的I分量到其他二分量上
# 使用cv2.merge
res1 = cv.merge([H, S, equ_I])
# 使用numpy的dstack()堆叠分量
# res = np.dstack((H, S, equ_I))
# 水平叠加数组，以便显示在一个视窗里对比
comp_res1 = np.hstack((hsi_img, res1))
cv.imshow('[H,S,equ_I]res_comparison[Left: HSI_img|Right: equI_img]', comp_res1)

# 对原图的BGR分量每个都做均衡化处理
(B, G, R) = cv.split(img)
(equ_B, equ_G, equ_R) = map((lambda img_weight: cv.equalizeHist(img_weight)), (B, G, R))
res2 = cv.merge([equ_B, equ_G, equ_R])
comp_res2 = np.hstack((img, res2))
cv.imshow('equ[B, G, R]res_comparison[Left: BGR_img|Right: equBGR_img]', comp_res2)

cv.waitKey(0)
cv.destroyAllWindows()
