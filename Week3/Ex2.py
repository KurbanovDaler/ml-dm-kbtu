import numpy as np
import random
import matplotlib.pyplot as plt
import pandas

def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a-b) <= (atol + rtol * np.abs(b))

def sigma(z):
	return 1/(1 + np.exp(-z))

# m = 100
# X1 = np.random.randint(1000, size=m).astype(dtype=np.float32)
# X2 = np.random.randint(1000, size=m).astype(dtype=np.float32)
# Y = np.array([1 if x2 > f(x1) else 0
# 			for x1, x2 in zip(X1, X2)], dtype=np.float32)
# for i in range(m):
# 	if random.random() < 0.03:
# 		Y[i] = 1 - Y[i] 
data = pandas.read_csv('ex2data1.txt')
data = data.to_numpy()

y = [data[i][2] for i in range(len(data))]
x = np.delete(data, 2, 1)

# y = [(i-np.mean(y))/(np.max(y)-np.min(y)) for i in y]

# x = np.array([[(x[i][j]-np.mean(x[:, j]))/(np.max(x[:, j])-np.min(x[:, j])) for j in range(len(x[i]))] for i in range(len(x))])

x = np.array([np.insert(i, 0, 1) for i in x])


# print(y)
# exit(0)
theta = np.array([0., 0., 0.])
lr = 1.
m = len(x)
dw_s=0
for _ in range(100000):
	Z = x.dot(theta)
	a = sigma(Z)
	dz = a - y
	dw = x.T.dot(dz)/m
	dw_s = dw_s*0.9 + 0.1*dw
	theta -= lr*dw_s

print(theta)

aa = -theta[1]/theta[2]
bb = -theta[0]/theta[2]

def g(x):
	return aa*x + bb

print(y)

yy = np.ones(m)
red = np.where(isclose(y, yy))
yy = np.zeros(m)
blue = np.where(isclose(y, yy))

print(x)

plt.plot(x[red, 1], x[red, 2], 'ro')
plt.plot(x[blue, 1], x[blue, 2], 'bo')
# for i in range(len(x)):

    

_x = [np.amin(x[:, 1]), np.amax(x[:, 2])]
_y = [g(i) for i in _x]
plt.plot(_x, _y)

plt.show()
