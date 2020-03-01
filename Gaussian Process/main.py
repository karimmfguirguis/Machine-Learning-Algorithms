import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
from mpl_toolkits.mplot3d import Axes3D

# ==================================================================#
# Function definitions                                              #
# ==================================================================#


def gaussian(x, mean, cov):
    det = np.linalg.det(cov)
    constant = 1/(2*math.pi * np.sqrt(det))
    bracket = -0.5 * np.matmul(np.matmul((x-mean), np.linalg.inv(cov)), (x-mean).T )
    return constant*np.exp(bracket)


def kernel(x1, x2):
    return np.exp(-np.power((x1-x2)/0.2, 2))


def kernel1(x1, x2):
    return np.exp(-np.abs((x1-x2)/0.2))

# ==================================================================#
# Initialization                                                    #
# ==================================================================#
sigma = 0.1
lamda = 0.01
x_test = [i/100 for i in range(-100, 100, 2)]
x_test = np.asarray(x_test)
x = [-0.5, 0.5]
y = [0.3, -0.1]

# ==================================================================#
# Mean and covariance                                               #
# ==================================================================#
K = [[kernel(element1, element2) for element2 in x]for element1 in x]
likelihoodCov = np.linalg.inv(K + lamda * np.identity(2))
Ktest = [[kernel(element, x[i]) for i in range(2)] for element in x_test]
Ktest = np.asmatrix(Ktest)
mean = np.dot(np.dot(Ktest, likelihoodCov), y)
variance = [ (kernel(x_test[i], x_test[i]) - np.dot(np.dot(Ktest[i, :], likelihoodCov), Ktest[i, :].T)) for i in range(100) ]
standardDeviation = np.sqrt(np.asmatrix(np.ndarray.flatten(np.asarray(variance))))
varianceNoise = [ (kernel(x_test[i], x_test[i]) - np.dot(np.dot(Ktest[i, :], likelihoodCov), Ktest[i, :].T) + sigma) for i in range(100) ]
standardDeviationNoise = np.sqrt(np.asmatrix(np.ndarray.flatten(np.asarray(varianceNoise))))

# ==================================================================#
# Plotting                                                          #
# ==================================================================#
x_test = np.asmatrix(x_test)
plt.scatter(x, y, color="black")
plt.plot(x_test.T, mean.T, color="red")
plt.plot(x_test.T, (mean + standardDeviation).T, color="gray")
plt.plot(x_test.T, (mean - standardDeviation).T, color="gray")
plt.plot(x_test.T, (mean + standardDeviationNoise).T, color="yellow")
plt.plot(x_test.T, (mean - standardDeviationNoise).T, color="yellow")
plt.show()
