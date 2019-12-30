# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 20:27:33
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 21:36:53
# 实验6 形态学图像处理
# 6.1 形态学转换
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

j = cv.imread('./exp.list/j.png', 0)
j1 = cv.imread('./exp.list/j1.png', 0)
j2 = cv.imread('./exp.list/j2.png', 0)
kernel = np.ones((5, 5), np.uint8)

# 6.1.1腐蚀, 让核模板在图象上滑动，如果与核模板元素对应的原图像的所有像素值为1，中心像素值不变;否则置0
erosion = cv.erode(j1, kernel, iterations=1)

# 6.1.2膨胀, 只要存在一个与核模板元素对应的一个像素值为1，就将中心像素值置1
dilation = cv.dilate(j2, kernel, iterations=1)

# 6.1.3开运算, 先腐蚀后膨胀就叫做开运算。它被用来去除噪声。用到的函数是cv.morphologyEx()
opening = cv.morphologyEx(j1, cv.MORPH_OPEN, kernel)

# 6.1.4闭运算, 先膨胀再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点。
closing = cv.morphologyEx(j2, cv.MORPH_CLOSE, kernel)

# 6.1.5形态学梯度 g = (f⊕b) - (f⊙b)
gradient = cv.morphologyEx(j, cv.MORPH_GRADIENT, kernel)

# 对比显示
titles = ['J with noise', 'Erosion', 'Opening', 'J with holes', 'Dilation', 'Closing', 'J', 'Gradience']
images = [j1, erosion, opening, j2, dilation, closing, j, gradient]
for i in range(len(titles)):
    plt.subplot(3, 3, i + 1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# 6.1.6边缘检测
# 形态学检测边缘的原理很简单
# 膨胀时，图像中的物体会想周围“扩张”；腐蚀时，图像中的物体会“收缩”。
# 比较这两幅图像，由于其变化的区域只发生在边缘, 所以将两幅图像相减，得到的就是图像中物体的边缘。
img = cv.imread("./exp.list/house.tif", 0)
# 构造一个3×3的结构元素
element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilate = cv.dilate(img, element)
erode = cv.erode(img, element)

# 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
result = cv.absdiff(dilate, erode)

# 上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
retval, result = cv.threshold(result, 40, 255, cv.THRESH_BINARY)
# 反色，即对二值图每个像素取反
result = cv.bitwise_not(result)
# 显示对比图像
comp_res = np.hstack((img, result))
cv.imshow("Comparison", comp_res)
cv.waitKey(0)
cv.destroyAllWindows()
