import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------------------------- #
#  		          Read input data from text file into dataMatrix (Nx3)            #
# ------------------------------------------------------------------------------- #
matrixData= np.loadtxt("..\..\..\Exercises\data2Class.txt")
N= np.shape(matrixData)[0] #number of training points
K= 3 #size of data
fig = plt.figure()
ax= fig.add_subplot(2,1,1)

# ------------------ #
#     Plot I/P       #
# ------------------ #
for row in matrixData:
    if row[2]==1.0:
        ax.plot(row[0],row[1], 'b+')
    else:
        ax.plot(row[0], row[1], 'ro')
ax.grid()

# -------------------------------------- #
#			   Linear features  		 #
# -------------------------------------- #
beta= np.zeros((K))  #initialize with zeros
X = matrixData[:, [0,1]]
ones = np.ones((N,1))
X = np.concatenate((ones,X), axis=1)
y = matrixData[:, 2]
lamda = 100000
Ik = np.identity(K)
p = np.zeros(N)
W = np.zeros((N,N))

# -------------------------------------------- # 
# 		   Logistic regression algorithm       #
# -------------------------------------------- # 
for iterator in range(10):
    for i in range(N):
        discrimFunc= np.dot(beta, X[i,:])
        p[i]= np.exp(discrimFunc) / (1+ np.exp(discrimFunc))
        W[i,i]= p[i]*(1-p[i])
    gradient= np.matmul(X.T,(p-y)) + 2*lamda*beta
    laplacian= np.matmul(np.matmul(X.T, W),X) + 2*lamda*Ik
    beta= beta- np.matmul(np.linalg.inv(laplacian), gradient)

# -------------------------------------------- #
# 				Plot decision boundary 		   #
# -------------------------------------------- #
x= np.linspace(-2.5,2.5,100)
decBoundary= -1/beta[2]*(beta[0] + beta[1]*x)
ax.plot(x,decBoundary, label= " linear decision boundary",color='green')
print("Beta= ", beta)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Logistic regression')


# -------------------------------------- #
#			   Quadratic features  		 #
# -------------------------------------- #
K=6
Ik= np.identity(K)
betaQuadratic= np.zeros((K))  #initialize with zeros
X= np.empty((N,K))
X[:,[1,2]]= matrixData[:, [0,1]]
lamda= 1
Ik= np.identity(K)
for j in range(N):
    X[j,0]= 1.0
    X[j,3]= np.square(matrixData[j,0])
    X[j,4]= matrixData[j,0]*matrixData[j,1]
    X[j,5]=np.square(matrixData[j,1])
y= matrixData[:, 2]
lamda= 0.01
Ik= np.identity(K)
p= np.zeros(N)
W= np.zeros((N,N))

# -------------------------------------------- # 
# 		   Logistic regression algorithm       #
# -------------------------------------------- # 
for iterator in range(10):
    for i in range(N):
        discrimFunc= np.dot(betaQuadratic, X[i,:])
        p[i]= np.exp(discrimFunc) / (1+ np.exp(discrimFunc))
        W[i,i]= p[i]*(1-p[i])
    gradient= np.matmul(X.T,(p-y)) + 2*lamda*betaQuadratic
    laplacian= np.matmul(np.matmul(X.T, W),X) + 2*lamda*Ik
    betaQuadratic= betaQuadratic- np.matmul(np.linalg.inv(laplacian), gradient)
print("Beta2= ",betaQuadratic)

# -------------------------------------------- #
# 				Plot decision boundary 		   #
# -------------------------------------------- #
x= np.linspace(-2.5,2.5,100)
y= np.linspace(-2.5,2.5,100)
X,Y= np.meshgrid(x,y)
Z= betaQuadratic[0]+ betaQuadratic[1]*X + betaQuadratic[2]*Y + betaQuadratic[3]*np.square(X) + betaQuadratic[4]*X*Y + betaQuadratic[5]*np.square(Y)
Sigmoid = np.exp(Z)/ (1 + np.exp(Z))
ax.contour(X,Y,Z,levels=[0], label= "Quadratic decision boundary", colors="black")
leg= ax.legend()

ax2= fig.add_subplot(2,1,2, projection='3d')
for row in matrixData:
    if row[2]==1.0:
        ax2.scatter(row[0],row[1],row[2], 'b+')
    else:
        ax2.scatter(row[0], row[1],row[2], 'ro')
ax2.plot_surface(X,Y,Sigmoid)
plt.show()
