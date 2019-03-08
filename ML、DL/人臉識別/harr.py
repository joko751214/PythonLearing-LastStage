import numpy as np
import cv2 as cv

# 級聯分類器:CascadeClassifier
face_detector = cv.CascadeClassifier('haar\\face.xml')
eye_detector = cv.CascadeClassifier('haar\\eye.xml')
nose_detector = cv.CascadeClassifier('haar\\nose.xml')

cap = cv.VideoCapture(0)
while True:
    image = cap.read()[1]
    image = cv.resize(image, None, fx=0.75, fy=0.75,
                      interpolation=cv.INTER_AREA)
    # 多尺度縮放檢測:detectMultiScale
    # 1.3:精度(精度過大,識別反而會抓不到輪廓), 5:範圍
    faces = face_detector.detectMultiScale(image, 1.3, 5)
    eyes = eye_detector.detectMultiScale(image, 1.3, 5)
    noses = nose_detector.detectMultiScale(image, 1.3, 5)
    # 左,頂,寬,高
    for l, t, w, h in faces:
        # a,b代表圓的中心位置
        a, b = int(w / 2), int(h / 2)
        # 畫一個橢圓:ellipse
        # 中心位置:(l + a, t + b), 水平和垂直半徑:(a, b)
        # 旋轉角度:0, 起始角度:0, 終止角度:360
        # 顏色參數:(255, 0, 255), 線條粗細:2
        cv.ellipse(image, (l + a, t + b), (a, b), 0, 0, 360,
                   (255, 0, 255), 2)
    for l, t, w, h in eyes:
        a, b = int(w / 2), int(h / 2)
        cv.ellipse(image, (l + a, t + b), (a, b), 0, 0, 360,
                   (255, 255, 0), 2)
    for l, t, w, h in noses:
        a, b = int(w / 2), int(h / 2)
        cv.ellipse(image, (l + a, t + b), (a, b), 0, 0, 360,
                   (0, 255, 255), 2)
    cv.imshow('VideoCapture', image)
    if cv.waitKey(33) == 27:
        break
cap.release()
cv.destroyAllWindows()
