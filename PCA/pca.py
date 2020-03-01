import matplotlib.pyplot as plt
import os
import math
import numpy as np
import numpy.linalg as sla
import scipy.sparse.linalg as sla
import scipy.misc as sp

#================================================================== #
# Load all images in matrix X(165, 77760)							#	
#================================================================== #
N= 165
d= 77760
X = np.zeros((N , d))
i=0
directoryPath= "C:\\Users\\George\\Documents\\Term 2 Masters\\Machine learning\\Exercises\\yalefaces"
for filename in os.listdir(directoryPath):
    fname = os.path.join(directoryPath, filename)
    X[i,:] = plt.imread(fname).flatten()
    i = i+1

#================================================================== #
# Compute mean and center data										#
#================================================================== #
mu = np.zeros((1,d))
for row in X:
    mu = np.add(mu, row)
mu = np.divide(mu, N)
centeredX = np.zeros((N , d))
j = 0
for row in X:
    centeredX[j,:]=row - mu
    j = j +1


#================================================================== #
# Singular Value decomposition of the centered Data                 #
#================================================================== #
p = 10
u, s, vt = sla.svds(centeredX, k=p)
Vp = vt.T
Z = np.matmul(centeredX, Vp)

#================================================================== #
# Reconstruction													#
#================================================================== #
newX = np.zeros((N , d))
newX= np.matmul(Z,Vp.T)
j = 0
for row in newX:
    newX[j,:] = row + mu
    j = j +1
print(newX.shape)
print(Vp.shape)

#================================================================== #
# Save images														#
#================================================================== #
for i in range(0,N):
    image = np.reshape(newX[i,:], (243,320))
    sp.imsave("Image %i.gif" % i,image)

for i in range(0 , p):
    image = np.reshape(Vp[:,i], (243,320))
    sp.imsave("V%i.gif" % i,image)

image = np.reshape(mu, (243,320))
sp.imsave("Mean.gif",image)


#================================================================== #
# Error calculation													#
#================================================================== #
error =0
for i in range(0, N-1):
    error = error + np.square(np.linalg.norm((X[i,:]-newX[i,:])))

print("Error = ", error)
