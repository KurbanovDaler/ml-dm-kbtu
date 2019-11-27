import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import pandas
import cv2
import tkinter
from copy import deepcopy
from scipy.io import loadmat
# matplotlib.use('TkAgg')

def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a-b) <= (atol + rtol * np.abs(b))

def sigma(z):
	return 1/(1 + np.exp(-z))

def mapFeatureForPlotting(X1):
    degree = 8
    out = np.ones(1)
    for i in range(1, degree + 1):        
        out = np.hstack((out, np.power(X1, i)))
    return out
def mapFeature(X1):
    degree = 8
    out = np.ones(len(X1))[:,np.newaxis]
    for i in range(1, degree + 1):
        out = np.hstack((out, np.power(X1, i)[:,np.newaxis]))
    return out

data=loadmat('ex5data1') 
# print(data)
x = np.array(data['X'])
y = np.array(data['y'].flatten())
x_val = np.array(data['Xval'])
y_val = np.array(data['yval'].flatten())
x_test = np.array(data['Xtest'])
y_test = np.array(data['ytest'].flatten())
m = len(x)
y = [(i-np.mean(y))/(np.max(y)-np.min(y)) for i in y]
y_val = [(i-np.mean(y_val))/(np.max(y_val)-np.min(y_val)) for i in y_val]
y_test = [(i-np.mean(y_test))/(np.max(y_test)-np.min(y_test)) for i in y_test]

x = np.array([[(x[i][j]-np.mean(x[:, j]))/(np.max(x[:, j])-np.min(x[:, j])) for j in range(len(x[i]))] for i in range(m)])
x_val = np.array([[(x_val[i][j]-np.mean(x[:, j]))/(np.max(x_val[:, j])-np.min(x_val[:, j])) for j in range(len(x_val[i]))] for i in range(len(x_val))])
x_test = np.array([[(x_test[i][j]-np.mean(x_test[:, j]))/(np.max(x_test[:, j])-np.min(x_test[:, j])) for j in range(len(x_test[i]))] for i in range(len(x_test))])

x = np.array([np.insert(i, 0, 1) for i in x])
x_val = np.array([np.insert(i, 0, 1) for i in x_val])
x_test = np.array([np.insert(i, 0, 1) for i in x_test])

x = mapFeature(x[:, 1])
x_val = mapFeature(x_val[:, 1])
x_test = mapFeature(x_test[:, 1])
# print(x)
# exit(0)
theta = np.ones(9)
lr = 0.001
rl = 1
alpha = 0.7
dw_s=0
print(x)
def cost_function(X, y, thetas):
    a = X.dot(theta.T)
    return np.sum((np.square(a - y))/(2 * m)) + rl / (2 * m) * np.sum(np.square(theta))

print(cost_function(x, y, theta))

print(theta)
costs = []
val_costs = []
# costs.append(cost_function(x, y, theta))
epochs = 5000
for _ in range(epochs):
    a = x.dot(theta.T)
    dt = x.T.dot(a - y)/m + rl / m * np.sum(theta[1:])
    theta = theta - dt
    costs.append(cost_function(x, y, theta))
    val_costs.append(cost_function(x_val, y_val, theta))

xcosts = range(epochs)
val_costs = range(epochs)
plt.plot(xcosts, costs, 'r')
plt.plot(xcosts, val_costs, 'b')
plt.show()
print(theta)

plt.plot(x[:,1], y, 'ro')
_x = np.arange(-2, 2, 0.01)
_y = [mapFeatureForPlotting(i).dot(theta.T) for i in _x]

plt.plot(_x, _y)
xmax = np.amax(x)
ymax = np.amax(y)
xmin = np.amin(x)
ymin = np.amin(y)

plt.axis([xmin, xmax, ymin - 1, ymax+ 1])
plt.show()