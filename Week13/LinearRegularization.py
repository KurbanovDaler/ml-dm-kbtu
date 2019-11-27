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

def mapFeature(X, p):
    m = len(X)
    X_poly = np.zeros((m, p))
    X_poly[:, 0] = X
    for i in range(1, p):
        X_poly[:, i] = X * X_poly[:, i-1]
    return X_poly

data=loadmat('ex5data1') 
x = np.array(data['X'].flatten())
y = np.array(data['y'].flatten())

x_train = deepcopy(x)
y_train = deepcopy(y)

x_val = np.array(data['Xval'].flatten())
y_val = np.array(data['yval'].flatten())
x_test = np.array(data['Xtest'].flatten())
y_test = np.array(data['ytest'].flatten())

m = len(x)

x = mapFeature(x, 8)
x_val = mapFeature(x_val, 8)
x_test = mapFeature(x_test, 8)


x = [[(x[i][j]-x[:, j].mean())/x[:, j].std() for j in range(len(x[i]))] for i in range(len(x))]
x_val = [[(x_val[i][j]-x_val[:, j].mean())/x_val[:, j].std() for j in range(len(x_val[i]))] for i in range(len(x_val))]
x_test = [[(x_test[i][j]-x_test[:, j].mean())/x_test[:, j].std() for j in range(len(x_test[i]))] for i in range(len(x_test))]

x = np.array([np.insert(i, 0, 1) for i in x])
x_val = np.array([np.insert(i, 0, 1) for i in x_val])
x_test = np.array([np.insert(i, 0, 1) for i in x_test])

theta = np.ones(9)
lr = 0.001
alpha = 0.001
def cost_function(X, y, thetas, rl):
    m = len(y)
    a = X.dot(theta.T)
    return np.sum(np.square(a - y))/(2 * m) + rl / (2 * m) * np.sum(np.square(theta[1:]))

def gradient(x, y, theta, rl):
    a = x.dot(theta.T)
    dt = x.T.dot(a - y)/m + theta.dot(rl / m) #rl / m * np.sum(theta[1:])
    dt[0] = x.T.dot(a - y)[0]/m
    return dt

m = len(y)
costs = np.zeros(m)
val_costs = np.zeros(m)
for i in range(m):
    iterations = 5000
    theta = np.zeros(9)
    
    for j in range(iterations):
        new_theta = gradient(x[0:i+1, :], y[0:i+1], theta, 0)
        theta -= alpha * new_theta
    costs[i] = cost_function(x[0:i+1, :], y[0:i+1], theta, 0)
    val_costs[i] = cost_function(x_val, y_val, theta, 0)

plt.plot(np.arange(1, m+1), costs, label = "Train")
plt.plot(np.arange(1, m+1), val_costs, label = "Val")

plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()
print(theta)
rls = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
errors = []
val_errors = []
for rl in rls:
    iterations = 10000
    theta = np.zeros(9)
    for j in range(iterations):
        new_theta = gradient(x, y, theta, rl)
        theta -= alpha * new_theta
    errors.append(cost_function(x, y, theta, rl))
    val_errors.append(cost_function(x_val, y_val, theta, rl))
    

plt.plot(rls, errors, 'r', label = "Train")
plt.plot(rls, val_errors, 'b', label = "Train")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Selecting lambda')
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.show()
# rl = 100
# m = len(y)
# costs = np.zeros(m)
# val_costs = np.zeros(m)
# for i in range(m):
#     iterations = 5000
#     theta = np.zeros(9)
    
#     for j in range(iterations):
#         new_theta = gradient(x[0:i+1, :], y[0:i+1], theta, rl)
#         theta -= alpha * new_theta
#     costs[i] = cost_function(x[0:i+1, :], y[0:i+1], theta, rl)
#     val_costs[i] = cost_function(x_val, y_val, theta, rl)

# plt.plot(np.arange(1, m+1), costs, 'r', label = "Train")
# plt.plot(np.arange(1, m+1), val_costs, 'b', label = "Val")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.title('Learning curve for linear regression')
# plt.xlabel('Number of training examples')
# plt.ylabel('Error')
# plt.show()
# print(theta)
# plt.plot(x_train, y_train, 'rx')
# plt.plot(x_train, x.dot(theta.T), 'bo')
# plt.show()