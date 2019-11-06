import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import pandas
import cv2
import tkinter
from copy import deepcopy
from scipy.special import expit
from scipy.special import softmax as soft

# matplotlib.use('TkAgg')
def sigma(z):
    return 1.0/(1.0 + np.exp(-z))
def ReLU(z):
    return z
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()  
np.random.seed(42)
lr = 0.1
class NeuralNetwork:
    def __init__(self, x, y, hidden_layer_size):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], hidden_layer_size) * 0.1
        self.weights2   = np.random.rand(hidden_layer_size, 7) * 0.1
        self.y          = y
        self.output     = np.zeros(y.shape)

    

      
    def forward_propagation(self):
        self.layer1 = sigma(np.dot(self.input, self.weights1))
        self.output = sigma(np.dot(self.layer1, self.weights2))
        # print(np.dot(self.input, self.weights1))
        # print(np.exp(-np.dot(self.input, self.weights1)))
        # exit(0)
        # print(self.layer1)
        # print(self.output)
        # print(soft(self.output))
        # exit(0)
        return softmax(self.output)
        return self.output
    
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y -self.output) * self.output * (1 - self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y -self.output) * self.output * (1 - self.output), self.weights2.T) * self.layer1 * (1 - self.layer1))
        # print(self.y)
        # print(self.output)
        self.weights1 += d_weights1 * lr
        self.weights2 += d_weights2 * lr
        # exit(0)
    
    def train(self, X, y):
        self.output = self.forward_propagation()
        self.backprop()

def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a-b) <= (atol + rtol * np.abs(b))
    
data = pandas.read_csv('glass.csv')

data = data.sample(frac=1).reset_index(drop=True)

y_data = data['Type']
data = data.drop('Type', axis = 1)

data = data.to_numpy()
y_data = y_data.to_numpy()


one_hot_labels = np.zeros((len(y_data), 7))
y_data -= 1
for i in range(len(y_data)):
    one_hot_labels[i, y_data[i]] = 1
NN = NeuralNetwork(data, one_hot_labels, 10)
for i in range(10): 
    print ("for iteration # " + str(i) + "\n")
    NN.train(data, one_hot_labels)
# print(NN.weights1)
# print(NN.weights2)
res = NN.forward_propagation()
res = np.array(res).astype(dtype=float)
# print(res)
# exit(0)

print ("Input : \n" + str(data))
print ("Actual Output: \n" + str(y_data))
print ("Predicted Output: \n" + str(NN.forward_propagation()))

# y = [(i-np.mean(y))/(np.max(y)-np.min(y)) for i in y]

# x = np.array([[(x[i][j]-np.mean(x[:, j]))/(np.max(x[:, j])-np.min(x[:, j])) for j in range(len(x[i]))] for i in range(len(x))])
