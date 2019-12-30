# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 14:22:16
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 14:36:51
# 实验5.4 图像匹配
# 使用模板匹配在一幅图像中查找目标，且会使用多对象的模板匹配
# 使用cv.matchTemplate()使用模板图像在输入图像上滑动，比饺子区域
# 返回值为灰度图像，每个像素点处的值表示匹配程度

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('./exp.list/lena.jpg', 0)
img2 = img.copy()
template = cv.imread('./exp.list/eye.png', 0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for i, meth in enumerate(methods):
    img = img2.copy()
    # exec 语句用来执行储存在字符串或文件中的 Python 语句。
    # 例如，我们可以在运行时生成一个包含 Python 代码的字符串，然后使用 exec 语句执行这些语句。
    # eval 语句用来计算存储在字符串中的有效 Python 表达式, 解析为变量
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # 使用不同的比较方法，对结果的解释不同
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # 找到匹配区域的左上角和右下角后， 建立目标矩形区域，设定边框为白色(255), 粗度(2)
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(2, 6, i + 1), plt.imshow(res, cmap='gray')
    plt.title(meth + '\nMatching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 6, i + 7), plt.imshow(img, cmap='gray')
    plt.title('Detected Area'), plt.xticks([]), plt.yticks([])

plt.show()
