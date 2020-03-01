import numpy as np
import scipy.misc as sp
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.image as mpimg
import math


# ==================================================================
# Convert images to greyscale
# ==================================================================
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
	
# ==================================================================
# Load all images in matrix X(135, 77760)
# ==================================================================
N = 135
d = 38880
X = np.zeros((N, d))
i = 0
dataPath = "..\..\..\Data"
for filename in os.listdir(dataPath):
    fname = os.path.join(dataPath, filename)
    img = mpimg.imread(fname)
    gray = rgb2gray(img)
    X[i, :] = gray.flatten()
    i = i + 1

# ==================================================================
# Initialization of mean vector(random values)
# ==================================================================
error = []
# for K in range(4,16):
K = 4
mean = np.zeros((K, d))
for i in range(0, K):
    mean[i, :] = X[np.random.randint(0, N-1),:]
Rnk = np.zeros((N, K))
J = 0

# ==================================================================
# K-clustering
# ==================================================================

for iterator in range(0, 10):
    # find rnk
    for n in range(0, N):
        dataSubMean = [np.dot((X[n, :] - mean[k, :]), (X[n, :] - mean[k, :])) for k in range(0, K)]
        Rnk[n, np.argmin(dataSubMean)] = 1
    # find mean
    for k in range(0, K):
        denominator = sum(Rnk[:, k])
        numerator = np.zeros((1, d))
        for n in range(0, N):
            numerator = numerator + Rnk[n, k] * X[n, :]
        mean[k, :] = numerator / denominator
    # calculate J
    J = 0
    for n in range(0, N):
        for k in range(0, K):
            J = J + Rnk[n, k] * np.dot((X[n, :] - mean[k, :]), (X[n, :] - mean[k, :]))
    print(J)


# ==================================================================
# Save samples
# ==================================================================
destPath = "..\..\src"
print(X.shape)
for k in range(0, K):
    folderName = "Cluster" + str(k)
    folderName = os.path.join(destPath, folderName)
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    sample = [i for i in range(0, N) if Rnk[i, k] > 0]
    for i in range(0, len(sample)):
        image = np.reshape(X[sample[i], :], (243, 160))
        fileName = "Sample" + str(i) + ".gif"
        fileName = os.path.join(folderName, fileName)
        sp.imsave(fileName, image)

# ==================================================================
# Save means
# ==================================================================
for k in range(0, K):
    image = np.reshape(mean[k, :], (243, 160))
    sp.imsave("Mean%i.gif" % k, image)
