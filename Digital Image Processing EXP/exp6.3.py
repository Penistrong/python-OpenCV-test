# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-27 21:17:51
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-27 21:48:57
# 颜色空间变换
import cv2 as cv
import numpy as np

# 使用摄像头取帧然后捕捉蓝色物体
cap = cv.VideoCapture(0)
# 设定蓝色的阈值
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
# 用于开运算和闭运算的核模板
kernel = np.ones((5, 5), np.uint8)
while(1):
    # 获取每一帧
    ret, frame = cap.read()
    # 开运算 消除噪音和空洞
    frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    # 转换到HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # 根据阈值构建掩模
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # 对原图像和掩模进行按位与运算
    res = cv.bitwise_and(frame, frame, mask=mask)
    # 显示对比图像
    output = np.hstack((frame, mask, res))
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
# 关闭窗口
cv.destroyAllWindows()
