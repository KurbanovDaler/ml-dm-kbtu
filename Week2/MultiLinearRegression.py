import numpy as np
import matplotlib.pyplot as plt
import pandas
from mpl_toolkits.mplot3d import Axes3D

f = open('ex1data2.txt')

line = f.readline()

X = []
Y = []
Z = []

while line:

    lines = line.split(',')

    # input.append([lines[0], lines[1], lines[2]])

    X.append([lines[0], lines[1]])
    # Y.append(lines[1])
    Z.append(lines[2])
    
    line = f.readline()

f.close()

# X = np.ndarray(shape=(len(Y), 2))

# for i in range(len(Y)):
#     X[i][0] = Y[i][0]
#     X[i][1] = Y[i][1]


ax = plt.figure().gca(projection='3d')

X = np.array(X).astype(np.float)
Z = np.array(Z).astype(np.float)

Xmax = np.amax(X, axis=0)
Xmin = np.amin(X, axis=0)

Zmax = np.amax(Z)
Zmin = np.amin(Z)
Xmean = np.mean(X, axis = 0)

print(Xmean[0], Xmean[1])
print(Xmax[0] - Xmin[0])
print(Xmax[1] - Xmin[1])
for i in range(len(X)):
    # X[i][0] /= Xmax[0] - Xmin[0]
    # X[i][1] /= Xmax[1] - Xmin[1]
    # Z[i] /= Zmax - Zmin
    # print(X[i][0])
    X[i][0] = (X[i][0] - Xmean[0]) / (Xmax[0] - Xmin[0])
    X[i][1] = (X[i][1] - Xmean[1]) / (Xmax[1] - Xmin[1])
    Z[i] = (Z[i] - np.mean(Z)) / (Zmax - Zmin)

ax.scatter(X[:,0], X[:,1], Z, 'rx')

plt.show()

lr = 0.01

costs = []

theta = np.array([0., 0., 0.])

def h(theta, x):
    return theta[0] + theta[1] * x[0] + theta[2] * x[1]

def cost(theta):
    result = 0.0
    for i in range(len(X)):
    		result += (h(theta, X[i]) - Z[i])**2
    result = result / (2*len(X))
    return result

print(cost(theta))

for i in range(1500):    
    dt0 = 0.0
    dt1 = 0.0
    dt2 = 0.0
    for j in range(len(X)):        
        dt0 += h(theta, X[j]) - Z[j]
        dt1 += (h(theta, X[j]) - Z[j]) * X[j][0]
        dt2 += (h(theta, X[j]) - Z[j]) * X[j][1]
    dt0 /= len(X)
    dt1 /= len(X)
    dt2 /= len(X)
    
    # print(dt0)
    # print(dt1)

    theta[0] = theta[0] - lr * dt0
    theta[1] = theta[1] - lr * dt1
    theta[2] = theta[2] - lr * dt2

    # print('-----------')

    # print(theta[0])
    # print(theta[1])
    # print('\n')


    costs.append(cost(theta))

print(cost(theta))
print(theta)

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

x = range(len(costs))

plt.plot(x, costs)

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], Z)

point  = np.array([0.0, 0.0, theta[0]])
normal = np.array(cross([1,0, theta[2]], [0,1, theta[1]]))
d = -point.dot(normal)
xx, yy = np.meshgrid([0,1], [0,1])
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
ax.plot_surface(xx, yy, z, alpha=0.2, color=[0,1,0])

plt.show()
# x = np.zeros(shape=(len(X), 2))
# y = [h(theta, i) for i in x]

# xx, yy = np.meshgrid(range(47), range(47))

# ax = plt.figure().gca(projection='3d')

# ax.plot_surface(xx, yy, )

# # plt.show()

# # plt.plot(thetas0, thetas1)

# # plt.axis([-10, 10, -10, 10])

# # plt.show()


