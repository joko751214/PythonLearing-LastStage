import numpy as np
# 數據集模塊
import sklearn.datasets as sd
# 特性選擇器
import sklearn.feature_selection as fs
# 隨機森林模塊
import sklearn.ensemble as se
# 管線模塊
import sklearn.pipeline as sp
# 模型選擇器
import sklearn.model_selection as ms
import matplotlib.pyplot as mp
# 樣本生成器:samples_generator
# 製造分類樣本:make_classification
# 類別:n_informative,特性數:n_features,重複校正次數:n_redundant
x, y = sd.samples_generator.make_classification(
	n_informative=4, n_features=20, n_redundant=0,
	random_state=5)
# k:選擇5個最好的
skb = fs.SelectKBest(fs.f_regression, k=5)
# 隨機森林分類器
rfc = se.RandomForestClassifier(n_estimators=25,
	max_depth=4)
model = sp.Pipeline([
	('selector', skb), ('classifier', rfc)])
print(ms.cross_val_score(model, x, y, cv=10,
	scoring='f1_weighted').mean())
# 修改管線內的相關內容
model.set_params(selector__k=2,
	classifier__n_estimators=10)
print(ms.cross_val_score(model, x, y, cv=10,
	scoring='f1_weighted').mean())
model.fit(x, y)
selected_mask = model.named_steps['selector'].get_support()
selected_indices = np.arange(x.shape[1])[selected_mask]
x = x[:, selected_indices]
model.fit(x, y)
l, r, h = x[:, 0].min() - 1, x[:, 0].max() + 1, 0.005
b, t, v = x[:, 1].min() - 1, x[:, 1].max() + 1, 0.005
grid_x = np.meshgrid(np.arange(l, r, h), np.arange(b, t, v))
flat_x = np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
flat_y = model.predict(flat_x)
grid_y = flat_y.reshape(grid_x[0].shape)
mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
mp.title('Pipeline', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='Dark2')
mp.xlim(grid_x[0].min(), grid_x[0].max())
mp.ylim(grid_x[1].min(), grid_x[1].max())
mp.scatter(x[:, 0], x[:, 1], c=y, cmap='cool', s=80)
mp.show()
