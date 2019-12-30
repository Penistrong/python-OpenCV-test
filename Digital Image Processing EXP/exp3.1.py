# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-25 12:28:15
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-25 13:01:26

import cv2 as cv
import numpy as np

# 3.1缩放
img = cv.imread('./exp.list/flower2.jpg')
# 设置缩放因子(fx, fy)进行缩放，或者将None直接设置为输出的尺寸即可
resImg = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
#height, width=img.shape[:2]
#resImg = cv.resize(img, (2*width, 2*height), fx=2, fy=2, interpolation = cv.INTER_CUBIC)

# 3.2旋转
# 使用旋转矩阵，cv提供函数getRotationMatrix2D
img2 = cv.imread('./exp.list/lily.tif', 0)
rows, cols = img2.shape
# create Rotation Matrix
# param1:rotation center;param2:angle;param3:interpolation
Matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.6)
dst = cv.warpAffine(img2, Matrix, (cols, rows))

while(1):
    cv.imshow('img', img)
    cv.imshow('result', resImg)
    cv.imshow('rotation_result', dst)
    if cv.waitKey(1) & 0xFF == 27:
        break
cv.destroyAllWindows()
