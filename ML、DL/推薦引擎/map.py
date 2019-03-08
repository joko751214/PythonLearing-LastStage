import numpy as np


def f1(x):
    return x + 3


a = f1(1)
print(a)

b = [1, 2, 3, 4, 5, 6]
c = []
for x in b:
    c.append(f1(x))
print(c)

d = [f1(x) for x in b]
print(d)

# list:將map的返回值定義為list
# 在Python3中有惰性優化
# 惰性優化:因為map返回的是可迭代對象,若不迭代,系統就不會迭代
e = list(map(f1, b))
print(e)

f = f1(np.array(b)).tolist()
print(f)
