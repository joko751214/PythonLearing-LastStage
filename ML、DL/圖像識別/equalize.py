import numpy as np
import cv2 as cv

image = cv.imread('sunrise.jpg')
cv.imshow('Original', image)

# 如何將亮度單獨抽出來呢?
# 1.轉換成灰度圖
# 轉換顏色:cvtColor
# cv.COLOR_BGR2GRAY:將3通道突變成灰度
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 2.均衡化來提升亮度
# equalizeHist(gray):將gray做均衡化
equalized_gray = cv.equalizeHist(gray)
cv.imshow('Equalized Gray', equalized_gray)

# 如何將彩色圖提高亮度呢?
# cv.COLOR_BGR2YUV:轉換到yuv空間(亮度,色度,飽和度)
# yuv第三維的0列為亮度,1列為色度,2列為飽和度
yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
yuv[..., 0] = cv.equalizeHist(yuv[..., 0])
equalized_color = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
cv.imshow('Equalized Color', equalized_color)

cv.waitKey()
