import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import pandas
import cv2
from copy import deepcopy
from scipy.special import expit
from scipy.special import softmax as soft

# matplotlib.use('TkAgg')
def sigma(z):
    return 1.0/(1.0 + np.exp(-z))
def ReLU(z):
    return z
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()  
def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)
# np.random.seed(42)
lr = 0.01
costs = []
class NeuralNetwork:
    def __init__(self, x, y, hidden_layer_size):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], hidden_layer_size) #* 0.01
        self.weights2   = np.random.rand(hidden_layer_size, 7) #* 0.01
        self.bias1      = np.random.rand(10)
        self.bias2      = np.random.rand(7)
        self.y          = y
        self.output     = np.zeros(y.shape)

    def partial_derivative(self, x):
        return x * (1 - x)
    def forward_propagation(self):
        self.layer1 = sigma(np.dot(self.input, self.weights1) + self.bias1)
        self.output = softmax(np.dot(self.layer1, self.weights2) + self.bias2)
        # print(np.dot(self.input, self.weights1))
        # print(np.exp(-np.dot(self.input, self.weights1)))
        #exit(0)
        # print(self.layer1)
        # print(self.output)
        #exit(0)
        return self.output
    
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (self.output -self.y))
        tmp = np.dot(self.output - self.y, self.weights2.T)
        d_weights1 = np.dot(self.input.T, self.partial_derivative(self.layer1) * tmp)
        # d_weights1 = np.dot(self.input.T, np.dot((self.y -self.output) * self.output * (1 - self.output), self.weights2.T) * self.layer1 * (1 - self.layer1))
        # print(self.y)
        # print(self.output)
        self.weights1 -= d_weights1 * lr
        self.weights2 -= d_weights2 * lr
        self.bias1 -= (tmp * self.partial_derivative(self.layer1)).sum(axis=0) * lr
        self.bias2 -= (self.output - self.y).sum(axis=0) * lr
        # exit(0)
    
    def train(self, X, y):
        for i in range(35000): 
            self.output = self.forward_propagation()
            self.backprop()
            # if i % 100 == 0:
            #     print ("for iteration # " + str(i) + "\n")
            loss = np.sum(-self.y * np.log(self.output))    
            # print(loss)
            costs.append(loss)
            
            


def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a-b) <= (atol + rtol * np.abs(b))
    
data = pandas.read_csv('glass.csv')

data = data.sample(frac=1).reset_index(drop=True)

y_data = data['Type']
data = data.drop('Type', axis = 1)

data = data.to_numpy()
y_data = y_data.to_numpy()

data = np.array([[(data[i][j]-np.mean(data[:, j]))/(np.max(data[:, j])-np.min(data[:, j])) for j in range(len(data[i]))] for i in range(len(data))])


one_hot_labels = np.zeros((len(y_data), 7))
y_data -= 1
for i in range(len(y_data)):
    one_hot_labels[i, y_data[i]] = 1
NN = NeuralNetwork(data, one_hot_labels, 10)
print("smth")
NN.train(data, one_hot_labels)
# for i in range(1000): 
#     print ("for iteration # " + str(i) + "\n")
#     NN.train(data, one_hot_labels)
#     loss = np.sum(-self.one_hot_labels * np.log(normalized_probability_distribution))    
# print(NN.weights1)
# print(NN.weights2)
res = NN.forward_propagation()
res = np.array(res).astype(dtype=float)
print("WEIGHTS")
print(NN.weights1)
print(NN.weights2)
# # # exit(0)
result = np.argmax(res, axis = 1)
print(np.mean(result == y_data) * 100)
x = range(len(costs))
plt.plot(x, costs)
plt.savefig("costs.png")
# print ("Input : \n" + str(data))
# print ("Actual Output: \n" + str(y_data))
# print ("Predicted Output: \n" + str(NN.forward_propagation()))

# y = [(i-np.mean(y))/(np.max(y)-np.min(y)) for i in y]

# x = np.array([[(x[i][j]-np.mean(x[:, j]))/(np.max(x[:, j])-np.min(x[:, j])) for j in range(len(x[i]))] for i in range(len(x))])
