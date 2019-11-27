import numpy as np
import random
import matplotlib.pyplot as plt
import pandas

def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a-b) <= (atol + rtol * np.abs(b))

def sigma(z):
	return 1/(1 + np.exp(-z))

def mapFeatureForPlotting(X1, X2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
    return out
def mapFeature(X1, X2):
    degree = 6
    out = np.ones(len(X1))[:,np.newaxis]
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.hstack((out, np.multiply(np.power(X1, i -j), np.power(X2, j))[:,np.newaxis]))    
    return out

data = pandas.read_csv('ex2data2.txt')
data = data.to_numpy()

y = [data[i][2] for i in range(len(data))]
y = np.array(y)
x = np.delete(data, 2, 1)

# y = [(i-np.mean(y))/(np.max(y)-np.min(y)) for i in y]

# x = np.array([[(x[i][j]-np.mean(x[:, j]))/(np.max(x[:, j])-np.min(x[:, j])) for j in range(len(x[i]))] for i in range(len(x))])

x = np.array([np.insert(i, 0, 1) for i in x])
m = len(x)

x = mapFeature(x[:, 1], x[:, 2])
theta = np.zeros(28)
lr = 0.7
m = len(x)
dw_s=0

costs = []
def cost_function(X, y, theta):
    h = 1/(1 + np.exp(-X.dot(theta)))
    cost_1 = np.log(h)
    cost_2 = np.array([np.log(1-i) for i in h])
    summ = y.dot(cost_1) + np.array([1-i for i in y]).dot(cost_2)
    return -summ/m

costs.append(cost_function(x, y, theta))

print(cost_function(x, y, theta))
for i in range(200000):
    Z = x.dot(theta)
    a = sigma(Z)
    dz = a - y
    dw = x.T.dot(dz)/m
    dw_s = dw_s * 0.9 + 0.1 * dw
    theta -= lr * dw_s
    # costs.append(cost_function(x, y, theta))

print(theta)

yy = np.ones(m)m
red = np.where(isclose(y, yy))
yy = np.zeros(m)
blue = np.where(isclose(y, yy))

plt.plot(x[red, 1], x[red, 2], 'ro')
plt.plot(x[blue, 1], x[blue, 2], 'bo')

u = np.linspace(-1, 1.1, 50)
v = np.linspace(-1, 1.1, 50)
z = np.zeros((len(u), len(v)))

for i in range(len(u)):
    for j in range(len(v)):
        z[i,j] = np.dot(mapFeatureForPlotting(u[i], v[j]), theta)

pred = [sigma(np.dot(x, theta)) >= 0.5]
print(np.mean(pred == y.flatten()) * 100)

plt.contour(u,v,z,0)
plt.show()

# epochs = 200001
# X = range(epochs)

# plt.plot(X, costs)

# plt.ylabel('Cost')
# plt.xlabel('Epochs')

# plt.show()