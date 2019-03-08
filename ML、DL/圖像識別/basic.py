import numpy as np
import cv2 as cv

image = cv.imread('forest.jpg')
print(image.shape)	# (397, 600, 3) -> (高度,寬度,3個顏色通道(RGB))
print(image)
# 顯示圖片
cv.imshow('Original', image)
# 取出高度和寬度
h, w = image.shape[:2]
# 矩型的左上角為原點, x座標:l,y座標:t
l, t = int(w / 4), int(h / 4)
# 矩型的右下角, x座標:r,y座標:b
r, b = int(w * 3 / 4), int(h * 3 / 4)
# 從原圖擷取一段出來
cropped = image[t:b, l:r]
cv.imshow('Cropped', cropped)

blue = np.zeros_like(cropped)
green = np.zeros_like(cropped)
red = np.zeros_like(cropped)
# 將第三維的內容取出來, 取出第0列,而其他為0所以顯示藍色
blue[..., 0] = cropped[..., 0]
cv.imshow('Blue', blue)
# 第1列是綠色
green[..., 1] = cropped[..., 1]
cv.imshow('Green', green)
# 第2列是紅色
red[..., 2] = cropped[..., 2]
cv.imshow('Red', red)
# 縮放圖像:resize
# 若將cropped放大到跟原圖(w,h)一樣,則勢必會有一些像素的位置是空缺的
# 所以這時可以用:interpolation(平滑過渡的效果)
# cv.INTER_LINEAR,去計算相素間的差值來安插顏色
scaled = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)
cv.imshow('Scaled', scaled)
# fx=2, 水平方向的倍數;fy=0.5, 垂直方向的倍數
deformed = cv.resize(cropped, None, fx=2, fy=0.5,
                     interpolation=cv.INTER_LINEAR)
cv.imshow('Deformed', deformed)

# 等待鍵,直到敲任意鍵之後影像會消失
cv.waitKey()
