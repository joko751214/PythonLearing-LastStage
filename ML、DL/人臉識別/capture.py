import numpy as np
import cv2 as cv

# 視頻捕捉,參數0:代表只有一項設備
# 返回的是視頻捕捉對象
cap = cv.VideoCapture(0)
while True:
    image = cap.read()[1]
    # # interpolation:內差值
    image = cv.resize(image, None, fx=0.75, fy=0.75,
    			interpolation=cv.INTER_AREA)  
    cv.imshow('VideoCapture', image)
    # 33毫秒, 27:代表Esc鍵
    if cv.waitKey(33) == 27:
        break
# 釋放資源
cap.release()
# 關掉隱藏窗口
cv.destroyAllWindows()
