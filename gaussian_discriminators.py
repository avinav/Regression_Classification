import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    return    
    #return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    return
    #return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    
    return
    #return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    
    return
    #return acc
    
#------------------Script Starts Here------------------------------
# Load sample data
data = pickle.load(open("sample.pickle","rb"))
[data0, data1, data2, data3] = data

# Plot sample data 
plt.close("all")

colors = iter(cm.rainbow(np.linspace(0,1,5)))
plt.clf()
fig = plt.figure()
for i in range(1,6):
    plt.scatter(data0[(data1==i).reshape((data1==i).size),0],data0[(data1==i).reshape((data1==i).size),1],color=next(colors))
fig.suptitle('Sample data 1')
fig.show()

colors = iter(cm.rainbow(np.linspace(0,1,5)))
fig = plt.figure()
for i in range(1,6):
    plt.scatter(data2[(data3==i).reshape((data3==i).size),0],data2[(data3==i).reshape((data3==i).size),1],color=next(colors))
fig.suptitle('Sample data 2')
fig.show()

# Load diabetes data
dtdata = pickle.load(open('diabetes.pickle','rb'))
[dtdata0, dtdata1, dtdata2, dtdata3] = dtdata

# Low rank approximation of train data
U, s, Vh = linalg.svd(dtdata0.T)
temps = np.copy(s)
s [2:] = 0
sk = linalg.diagsvd(s, U.shape[1], Vh.shape[0])
dt0_app = np.dot(U, np.dot(sk, Vh))
dt0_app = dt0_app[:,:2]
fig = plt.figure()
plt.scatter(dt0_app[:,0],dt0_app[:,1])
fig.suptitle('Diabetes data in 2-D with low rank approximation')
fig.show()


