import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
from mpl_toolkits.mplot3d import Axes3D

# =========================================================== #
# 						Sigmoid function                      #
# =========================================================== #
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

# =========================================================== #
# 						Loss function                         #
# =========================================================== #
def loss(num1, num2):
    return np.maximum(0, 1 - num1 * num2)
	
# =========================================================== #
# 					Forward propagation function              #
# =========================================================== #
def forwardprop(x, w0, w1):
    z1 = np.matmul(w0, x)
    x1 = sigmoid(z1)
    f = np.matmul(w1, x1)
    return (f, x1, z1)



# =========================================================== #
# 					Backward propagation function             #
# =========================================================== #
def backwardprop(delta2, x0, x1, w1):
    delta1 = delta2 * w1
    delta1 = np.multiply(delta1, np.multiply(x1, 1-x1).T)
    gradW1 = delta2 * x1.T
    gradW0 = np.outer(delta1.T, x0.T)
    return (delta1, gradW0, gradW1)

# =========================================================== #
#			  Load data and initialize parameters             #
# =========================================================== #

fileName = "..\..\..\data2Class_adjusted.txt"
matrixData = np.loadtxt(fileName)
x0 = matrixData[:, [0, 1, 2]]
y = matrixData[:, 3]
N = 200
h0 = 3
h1 = 100
W0 = np.zeros((h1, h0))
W1 = np.zeros((1, h1))
alpha = 0.05

for i in range(h1):
    for j in range(h0):
        W0[i, j] = np.random.uniform(-1, 1)
        # W0[i, j] = 0

for k in range(h1):
    W1[0, k] = np.random.uniform(-1, 1)


# =========================================================== #
# 							Training                          #
# =========================================================== #
error = 100
loss_vector = []
iteration = 1
while(np.abs(error) > 1e-2):
    print("iteration", iteration)
    iteration = iteration + 1
    gradW0 = np.zeros((h1, h0))
    gradW1 = np.zeros((1, h1))
    lossValue = 0
    for point in range(N):
        f, x1, z1 = forwardprop(x0[point, :].T, W0, W1)
        delta2 = 0
        # print(loss(f, y[point]))
        lossValue = lossValue + loss(f, y[point])
        if (1 - f * y[point]) > 0:
            delta2 = - y[point]
        delta1_point, gradW0_point, gradW1_point = backwardprop(delta2, x0[point, :].T, x1, W1)
        gradW0 = gradW0 + gradW0_point
        gradW1 = gradW1 + gradW1_point
    # print("gradW0= ", gradW0)
    # print("gradW1= ", gradW1)
    W0_old = W0
    W1_old = W1
    W0 = W0 - alpha * gradW0
    W1 = W1 - alpha * gradW1
    error = np.sum(W0 - W0_old) + np.sum(W1 - W1_old)
    loss_vector.append(lossValue)
    print(lossValue)

print(error)
print("W0= ", W0)
print("W1= ", W1)

# =========================================================== #
# 							Plot                              #
# =========================================================== #
fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
for row in matrixData[:, [1, 2, 3]]:
    if row[2] == 1.0:
        ax.plot(row[0], row[1], 'b+')
    else:
        ax.plot(row[0], row[1], 'ro')
ax.grid()
ax2 = fig.add_subplot(3, 1, 2)
x = np.linspace(-2.5, 2.5, 100)
y = np.linspace(-2.5, 2.5, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        input_x= [[1], [x[j]], [y[i]]]
        Z[i][j], dummy1, dummy2 = forwardprop(input_x, W0, W1)
Z = sigmoid(Z)
ax.contour(X, Y, Z, levels=[0.5], label=" decision boundary")

ax2 = fig.add_subplot(2,1,2, projection='3d')
for row in matrixData:
    if row[3] == 1.0:
        ax2.scatter(row[1], row[2], row[3], 'b+')
    else:
        ax2.scatter(row[1], row[2], row[3], 'ro')
ax2.plot_surface(X, Y, 2*Z-1)

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(loss_vector)
plt.show()
