import numpy as np
import neurolab as nl
import matplotlib.pyplot as mp

n_samples = 30
x1 = np.cos(np.arange(0, n_samples)) * 1
x2 = np.cos(np.arange(0, n_samples)) * 2
x3 = np.cos(np.arange(0, n_samples)) * 3
x4 = np.cos(np.arange(0, n_samples)) * 4
train_x = np.array([x1, x2, x3, x4]).reshape(-1, 1)
y1 = np.ones(n_samples) * 0
y2 = np.ones(n_samples) * 4
y3 = np.ones(n_samples) * 2
y4 = np.ones(n_samples) * 0
train_y = np.array([y1, y2, y3, y4]).reshape(-1, 1)
# newel(輸入範圍,神經元數和輸出數,傳導函數)
# 傳導函數:傳導連接信號和回饋信號
model = nl.net.newelm([[-2, 2]], [10, 1],
                      [nl.trans.TanSig(), nl.trans.PureLin()])
# 對模型層做初始化
model.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
model.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
model.init()
error = model.train(train_x, train_y, epochs=1000, show=100,
                    goal=0.01)
pred_train_y = model.sim(train_x)
n_samples = 60
x1 = np.cos(np.arange(0, n_samples)) * 1
x2 = np.cos(np.arange(0, n_samples)) * 2
x3 = np.cos(np.arange(0, n_samples)) * 3
x4 = np.cos(np.arange(0, n_samples)) * 4
test_x1 = np.array([x1, x2, x3, x4]).reshape(-1, 1)
y1 = np.ones(n_samples) * 0
y2 = np.ones(n_samples) * 4
y3 = np.ones(n_samples) * 2
y4 = np.ones(n_samples) * 0
test_y1 = np.array([y1, y2, y3, y4]).reshape(-1, 1)
pred_test_y1 = model.sim(test_x1)
n_samples = 90
x1 = np.cos(np.arange(0, n_samples)) * 1
x2 = np.cos(np.arange(0, n_samples)) * 2
x3 = np.cos(np.arange(0, n_samples)) * 3
x4 = np.cos(np.arange(0, n_samples)) * 4
test_x2 = np.array([x1, x2, x3, x4]).reshape(-1, 1)
y1 = np.ones(n_samples) * 0
y2 = np.ones(n_samples) * 4
y3 = np.ones(n_samples) * 2
y4 = np.ones(n_samples) * 0
test_y2 = np.array([y1, y2, y3, y4]).reshape(-1, 1)
pred_test_y2 = model.sim(test_x2)
mp.figure().set_facecolor(np.ones(3) * 240 / 255)
mp.title('Recurrent Neural Networks', fontsize=20)
mp.xlabel('Time', fontsize=14)
mp.ylabel('Signal', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(train_y, c='dodgerblue', label='True')
mp.plot(pred_train_y, c='limegreen', label='Prediction')
mp.legend()
mp.figure().set_facecolor(np.ones(3) * 240 / 255)
mp.title('Recurrent Neural Networks', fontsize=20)
mp.xlabel('Time', fontsize=14)
mp.ylabel('Signal', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(test_y1, c='dodgerblue', label='True')
mp.plot(pred_test_y1, c='limegreen', label='Prediction')
mp.legend()
mp.figure().set_facecolor(np.ones(3) * 240 / 255)
mp.title('Recurrent Neural Networks', fontsize=20)
mp.xlabel('Time', fontsize=14)
mp.ylabel('Signal', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(test_y2, c='dodgerblue', label='True')
mp.plot(pred_test_y2, c='limegreen', label='Prediction')
mp.legend()
mp.figure().set_facecolor(np.ones(3) * 240 / 255)
mp.title('Training Progress', fontsize=20)
mp.xlabel('Number Of Epochs', fontsize=14)
mp.ylabel('Training Error', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(error, c='orangered', label='Error')
mp.legend()
mp.show()
