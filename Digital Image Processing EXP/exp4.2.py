# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 09:48:26
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 10:24:44
# 直方图均衡化 cv2.equalizeHist()
import cv2 as cv
import numpy as np

# 灰度图像均衡化
img = cv.imread('./exp.list/tyre.png')
# 转化为8位灰度图,如果在上面一行使用0作为第二参数，默认直接为灰度图
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
equ = cv.equalizeHist(img)
res = np.hstack((img, equ))
cv.imwrite('hist.jpg', res)
cv.imshow('Comparison', res)

cv.waitKey(0)
cv.destroyAllWindows()
