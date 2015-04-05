import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from matplotlib.patches import Ellipse
from pylab import savefig
from mpl_toolkits.mplot3d import Axes3D

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD                                                   
    id_mat = np.identity(X.shape[1])
    return np.dot(np.dot(linalg.inv(lambd*id_mat + np.dot(X.T, X)), X.T), y)

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1 (? N x 1)
    # Output:
    # rmse
    # IMPLEMENT THIS METHOD
    return np.sqrt(np.mean(np.square(ytest - np.dot(Xtest, w))))

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

#------------------Script starts here-----------------------

plt.close('all')    
k = 400 
lambdas = np.linspace(0, 1, num=k)
X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

#lambda_ = 0.4
#w_l = learnRidgeRegression(X_i,y,lambda_)
#rmse = testOLERegression(w_l,Xtest_i,ytest)
#print('RMSE value at ' + str(lambda_) + ': ' + str(rmse))
fig = plt.figure()
plt.plot(lambdas,rmses3)

fig.suptitle('Variation of RMSE with regularization paramter, lambda')
plt.xlabel('lambda ->')
plt.ylabel('RMSE ->')
fig.show()
savefig('results/ridge_regress_lambda.png',format='png')