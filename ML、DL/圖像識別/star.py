import numpy as np
import cv2 as cv

image = cv.imread('table.jpg')
cv.imshow('Original', image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
# 特徵檢測器:xfeatures2d.StarDetector_create()
detector = cv.xfeatures2d.StarDetector_create()
# 檢測:detect,返回的是位置跟矢量
keypoints = detector.detect(gray)
# 畫關鍵點:drawKeypoints
# 參數:(原圖,畫的內容,目標圖,標誌位)
# cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS:包含位置跟方向
cv.drawKeypoints(image, keypoints, image,
                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Star Keypoints', image)
cv.waitKey()
