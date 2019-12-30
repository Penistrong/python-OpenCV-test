# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-25 13:02:29
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-26 20:12:21

import cv2 as cv
import numpy as np

# 3.3仿射变换 原图像与输出图像中的平行线仍然保持平行
img = cv.imread('./exp.list/moon.jpg')
rows, cols, ch = img.shape
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
Matrix = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img, Matrix, (cols, rows))
cv.imshow('Input', img)
cv.imshow('Output', dst)
cv.imwrite('getAffineTransformImg.jpg', dst)
cv.waitKey(0)
cv.destroyAllWindows()
