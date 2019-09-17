import numpy as np
import matplotlib.pyplot as plt
import pandas
from mpl_toolkits.mplot3d import Axes3D

data = pandas.read_csv('train_data.csv')
data = data.to_numpy()
y = [data[i][4] for i in range(len(data))]
x = np.delete(data, 4, 1)


thetas = np.zeros(len(data[0]))
lr = 0.01
epochs = 1500

y = [(i-np.mean(y))/(np.max(y)-np.min(y)) for i in y]
x = np.array([[(x[i][j]-np.mean(x[:, j]))/(np.max(x[:, j])-np.min(x[:, j])) for j in range(len(x[i]))] for i in range(len(x))])
x = np.array([np.insert(i, 0, 1) for i in x])

def cost(x, y, thetas):
    h = x.dot(thetas)
    return np.sum(np.square(h - y)) / (2 * len(x))

costs = []

m = len(x)

for i in range(epochs):
    next_thetas = np.zeros(5).astype(float)
    for j in range(5):
        next_thetas[j] = thetas[j] - (lr / m) * \
        np.sum((x.dot(thetas) - y) * np.array(x[:, j]))
    thetas = next_thetas

    costs.append(cost(x, y, thetas))

print(costs)

X = range(epochs)

plt.plot(X, costs)

plt.ylabel('Cost')
plt.xlabel('Epochs')

plt.show()