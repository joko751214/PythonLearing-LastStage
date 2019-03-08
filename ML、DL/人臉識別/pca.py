import os
import sys
import numpy as np
# 將低維度的模塊
import sklearn.decomposition as dc


def make_data():
    a = np.random.normal(size=250)
    b = np.random.normal(size=250)
    c = 2 * a + 3 * b
    d = 4 * a - b
    e = c + 2 * d
    x = np.c_[d, b, e, a, c]
    return x


def train_model(x):
    model = dc.PCA()
    model.fit(x)
    return model


def reduce_model(model, n_components, x):
    # n_components:降低維度
    model.n_components = n_components
    # 重新訓練
    x = model.fit_transform(x)
    return x


def main(argc, argv, envp):
    x = make_data()
    print(x)
    model = train_model(x)
    # 說明重要性
    variances = model.explained_variance_
    print(variances)
    # 設定閥值
    threshold = 0.8
    # 取出大於閥值的重要性數據
    useful_indices = np.where(variances > threshold)[0]
    print(useful_indices)
    n_useful = len(useful_indices)
    print(n_useful)
    # 降低維度之後,出來的數據並不是從原數據直接取出
    # 而是再將其作調整之後呈現
    x = reduce_model(model, n_useful, x)
    print(x)
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
