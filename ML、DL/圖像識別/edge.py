import numpy as np
import cv2 as cv

# cv.IMREAD_GRAYSCALE:變成單通道而非3通道
# 變成一個二維數組
image = cv.imread('chair.jpg', cv.IMREAD_GRAYSCALE)
print(image.shape)
print(image.dtype)
print(image)
cv.imshow('Original', image)
# 將像素的顏色變成浮點型:cv.CV_64F
# 要得到水平方向:1, 不要垂直方向:0
# ksize:卷積和的大小,若為5,則是5行5列
# 矩陣卷積:對應元素相乘再相加後來替換原中心的數
# 覆蓋的矩陣與原先的矩陣做矩陣卷積後,依序替換原中心的數
# 邊緣的計算部分由於沒有數值即取0來計算
hor = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
cv.imshow('Hor', hor)
# 垂直方向看到的是水平邊緣
ver = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
cv.imshow('Ver', ver)
# 水平垂直邊緣
hor_ver = cv.Sobel(image, cv.CV_64F, 1, 1, ksize=5)
cv.imshow('Hor-Ver', hor_ver)

# Laplacian邊緣檢測
laplacian = cv.Laplacian(image, cv.CV_64F)
cv.imshow('Laplacian', laplacian)

# Canny邊緣檢測:
# 50:卷積和的大小, 240:模糊度(模糊度越大,噪聲越少)
canny = cv.Canny(image, 50, 240)
cv.imshow('Canny', canny)

cv.waitKey()
