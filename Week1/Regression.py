import numpy as np
import matplotlib.pyplot as plt
import pandas

f = open('ex1data1.txt')

unsorted =[]

line = f.readline()

while line:

    lines = line.split(',')
    
    tuple_input = (float(lines[0]), float(lines[1]))
    unsorted.append(tuple_input)
    
    line = f.readline()

f.close()

unsorted.sort()

print(x[0] for x in unsorted)

X = []
Y = []

for i in range(len(unsorted)):
    # print(unsorted[i])
    X.append(unsorted[i][0])
    Y.append(unsorted[i][1])

# exit(0)

# xlabel = np.array(unsorted[:0]).astype(np.float)
# ylabel = np.array(unsorted[:1]).astype(np.float)

plt.plot(X, Y, 'rx')

xmax = np.amax(X)
ymax = np.amax(Y)

xmin = np.amin(X)
ymin = np.amin(Y)

plt.axis([xmin, xmax, ymin, ymax])

plt.show()

# plt.plot(x, y, 'rx', 'MarkerSize', 10)


lr = 0.01
costs = []
thetas0 = []
thetas1 = []
theta = np.array([0.0, 0.0])
# theta = np.array([0., 0.])

def h(theta, x):
    return theta[0] + theta[1] * x

def cost(theta):
    result = 0.0
    for i in range(len(X)):
    		result += (h(theta, X[i]) - Y[i])**2
    # print ("SQUARE COST\n")
    # print (result)
    result = result / (2*len(X))
    return result


for i in range(1500):
    thetas0.append(theta[0])
    thetas1.append(theta[1])
    dt0 = 0.0
    dt1 = 0.0
    for j in range(len(X)):        
        dt0 += h(theta, X[j]) - Y[j]
        dt1 += (h(theta, X[j]) - Y[j]) * X[j]
    dt0 /= len(X)
    dt1 /= len(X)
    
    # print(dt0)
    # print(dt1)

    theta[0] = theta[0] - lr * dt0
    theta[1] = theta[1] - lr * dt1

    # print('-----------')

    # print(theta[0])
    # print(theta[1])
    # print('\n')


    costs.append(cost(theta))

print ("LEN = \n")
print (len(X))
print(cost(theta))
print(theta)
plt.plot(X, Y, 'rx')

x = range(len(X))
y = [h(theta, i) for i in x]

plt.plot(x, y)
plt.axis([xmin, xmax, ymin, ymax])

plt.show()

# plt.plot(thetas0, thetas1)

# plt.axis([-10, 10, -10, 10])

# plt.show()


