import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy.sparse.linalg as sla
import scipy.stats as norm
# ==================================================================#
# define gaussian function                                          #
# ==================================================================#
def gaussian(x,mean,cov):
    det= np.linalg.det(cov)
    constant= 1/(2*math.pi* np.sqrt(det))
    bracket= -0.5 * np.matmul( np.matmul((x-mean),np.linalg.inv(cov)), (x-mean).T )
    return constant*np.exp(bracket)



# ==================================================================#
# Load data into matrix and initialization of parameters            #
# ==================================================================#
fname = "..\..\mixture.txt"
X = np.loadtxt(fname)
N = 300
D = 2
K = 3
pi = np.zeros(K)
mean = np.zeros((K, 2))
cov = np.zeros((2*K, 2))
responsibilities = np.zeros((N, K))
for row in X:
    plt.scatter(row[0], row[1], color="black")

# ==================================================================#
# Initialization                                                    #
# ==================================================================#
#initialize pi vector uniformly
pi = [1/K for i in range(3)]
#randomly select data points as  means
for row in range(K):
    mean[row, :] = X[np.random.randint(0,N-1),:]
#initialize covariance matrices with data covariance
dataCovariance = np.matmul(X.T, X)
for k in range(K):
    cov[[2*k, 2*k+1], :] = np.identity(2)

# ==================================================================#
# Gaussian mixture algorithm                                        #
# ==================================================================#
for iterator in range(15):
    #==================================E-Step======================================
    for n in range(N):
        for k in range(K):
            num = pi[k] * gaussian(X[n, :], mean[k, :], cov[[2*k, 2*k+1], :])
            den = 0
            for j in range(K):
                den = den + pi[j] * gaussian(X[n, :], mean[j, :], cov[[2*j, 2*j+1], :])
            responsibilities[n, k] = num/den
    #==================================M-Step======================================
    Nk = np.sum(responsibilities, axis=0)
    # update pi
    pi = [ element/N for element in Nk]
    # update mean
    for k in range(K):
        value = np.zeros(D)
        for n in range(N):
            value = value + responsibilities[n, k]*X[n, :]
        mean[k, :] = value / Nk[k]
    # update covariance
    for k in range(K):
        valueMatrix = np.zeros((D,D))
        for n in range(N):
            valueMatrix = valueMatrix + responsibilities[n, k]*np.outer((X[n, :]-mean[k, :]), (X[n, :]-mean[k, :]))
        cov[[2*k,2*k+1],:]= np.divide(valueMatrix, Nk[k])
    # ==================================log-likelihood======================================
    logLikelihood = 0
    for n in range(N):
        inner = 0
        for k in range(K):
            inner = inner + pi[k] * gaussian(X[n,:], mean[k, :], cov[[2*k, 2*k + 1], :])
        logLikelihood = logLikelihood + np.log(inner)
    print(logLikelihood)
# ==================================================================#
# update the plot with contours of normal distributions             #
# ==================================================================#
    for row in X:
        plt.scatter(row[0], row[1], color="black")
    x,y = np.mgrid[-4:4:.01, -4:4:.01]
    pos = np.dstack((x,y))
    for k in range(K):
        rv = norm.multivariate_normal(mean[k, :], cov[[2*k,2*k+1],:])
        plt.contour(x,y,rv.pdf(pos))
    plt.waitforbuttonpress()
    plt.pause(0.05)
    plt.clf()
plt.show()
