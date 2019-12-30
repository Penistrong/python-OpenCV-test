# -*- coding: utf-8 -*-
# @Author: Penistrong
# @Date:   2019-10-28 10:12:55
# @Last Modified by:   Penistrong
# @Last Modified time: 2019-10-28 18:07:00
# 实验3思考题
# 鼠标点选四个点，组成多边形区域，并通过仿射变换变换为规整矩形
import cv2 as cv
import numpy as np

global img
global pt1, pt2
ptsCount = 0
ptsMax = 4
lsPtsChosen = []
tpPtsChosen = []

img = cv.imread('./exp.list/Fig6A.jpg')
img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
img2 = img.copy()


def on_mouse(event, x, y, flags, param):
    global img, img2, pt1, pt2
    global ptsCount, ptsMax
    global lsPtsChosen, tpPtsChosen
    img2 = img2.copy()

    if event == cv.EVENT_LBUTTONDOWN:
        ptsCount = ptsCount + 1
        if(ptsCount > ptsMax):
            ptsCount = 0
            tpPtsChosen = []
        pt1 = (x, y)
        cv.circle(img2, pt1, 10, (0, 255, 0), 2)
        lsPtsChosen.append([x, y])  # 存入后用于转换为darry
        tpPtsChosen.append((x, y))  # 用于画点的存储数组
        print('已添加点，位置(%d, %d)' % (x, y))
        # 将选择的点用直线连接以显示
        for i in range(len(tpPtsChosen) - 1):
            cv.line(img2, tpPtsChosen[i], tpPtsChosen[i + 1], (255, 255, 255), 2)
            if(ptsCount == ptsMax):
                cv.line(img2, tpPtsChosen[3], tpPtsChosen[0], (255, 255, 255), 2)

        cv.imshow('Select polylines', img2)
        if(ptsCount == ptsMax):
            transformROI()
            lsPtsChosen = []
            tpPtsChosen = []
            # 清空绘图轨迹
            img2 = img.copy()


def transformROI():
    mask = np.zeros(img.shape, np.uint8)
    # ROI中的多边形的四个顶点
    pts1 = np.float32([lsPtsChosen])
    print(pts1)
    mask = cv.polylines(mask, np.int32(pts1), True, (0, 255, 255))
    mask2 = cv.fillPoly(mask, np.int32(pts1), (255, 255, 255))
    cv.imshow('Progress', mask2)
    p1, p2, p3, p4 = pts1[0]
    print(p1)
    # pts1.reshape((-1, 1, 2))
    # src_Area = cv.polylines(img, [pts1], True, (0, 255, 255))
    # 根据第一个点和第二个点,p[0],p[1]对应x, y
    p3[0] = p1[0]
    if p2[1] > p4[1]:
        p2[1] = p4[1]
    else:
        p4[1] = p2[1]
    pts2 = np.float32([p1, p2, p3, p4])
    print(pts2)
    cols = abs(p1[0] - p3[0])
    rows = abs(p2[1] - p4[1])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (cols, rows))
    cv.imshow('Result', dst)


def main():
    global img
    title = 'Select polylines'
    cv.namedWindow(title)
    cv.setMouseCallback(title, on_mouse)
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
