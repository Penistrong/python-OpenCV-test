# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-26 20:13:16
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-26 20:18:26

# 透视变换，使用getPerspectiveTransform()创建透视变换矩阵，使直线变换后仍是直线，找四个点，任意三个不能共线
import cv2 as cv
import numpy as np

img = cv.imread('./exp.list/Fig6B.jpg')
img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
rows, cols, ch = img.shape
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
Matrix = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(img, Matrix, (200, 200))
cv.imshow('input', img)
cv.imshow('output', dst)
cv.imwrite('getPerspectiveTransformImg.jpg', dst)
cv.waitKey(0)
cv.destroyAllWindows()
