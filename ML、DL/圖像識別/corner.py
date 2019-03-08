import numpy as np
import cv2 as cv

image = cv.imread('box.png')
cv.imshow('Original', image)

# 變成灰度圖
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
# 稜角檢測:cornerHarris
# (圖片,寬度,高度(卷積和),經度)
# 精度越高,檢測時間越長,失誤率越低
corner = cv.cornerHarris(gray, 7, 5, 0.04)
# 1.加強顏色
corner = cv.dilate(corner, None)
# 2.設定閥值
threshold = corner.max() * 0.01
corner_mask = corner > threshold
# 大於閥值的點都換成紅色
image[corner_mask] = [0, 0, 255]
cv.imshow('Corner', image)
cv.waitKey()
