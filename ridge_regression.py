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
from linear_regression import testOLERegression

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD         
    N = X.shape[0]                                          
    id_mat = np.identity(X.shape[1])
    return np.dot(np.dot(linalg.inv(N*lambd*id_mat + np.dot(X.T, X)), X.T), y)

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                           
    N = X.shape[0]  
    error = np.sqrt(np.sum(np.square(ytest - np.dot(X, w))))/(2*N) + 0.5*lambd*np.dot(w.T,w)
    error_grad = np.dot(y.T,X)/(-2*N) + np.dot(w.T,np.dot(X.T,X))/N + lambd*w.T
    return error, error_grad

#------------------Script starts here-----------------------

plt.close('all')    
k = 200
lambdas = np.linspace(0, 0.004, num=k)
X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

i = 0
rmses3 = np.zeros((k,1))
w_mag = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    w_mag[i] = np.log(np.dot(w_l.T,w_l)[0,0])
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

minrmse = np.min(rmses3)
argminrmse = lambdas[np.argmin(rmses3)]
print "minimum rmse value, ridge regress: ", minrmse
print "minimum rmse for lamba:", argminrmse

#lambda_ = 0.4
#w_l = learnRidgeRegression(X_i,y,lambda_)
#rmse = testOLERegression(w_l,Xtest_i,ytest)
#print('RMSE value at ' + str(lambda_) + ': ' + str(rmse))
fig = plt.figure()
plt.plot(lambdas,rmses3)
fig.suptitle('Variation of RMSE with regularization paramter,\n min RMSE: '
+str(minrmse)+' at '+str(argminrmse))
plt.xlabel('lambda ->')
plt.ylabel('RMSE ->')
fig.show()
savefig('results/ridge_regress_lambda.png',format='png')

fig = plt.figure()
plt.plot(lambdas,w_mag)
fig.suptitle('Variation of ln(||W||) with regularization paramter')
plt.xlabel('lambda ->')
plt.ylabel('ln(||W||) ->')
fig.show()
savefig('results/ridge_regress_weight.png',format='png')

# Gradient Descent

k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
w_mag4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, X, y, lambd, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    w_mag4[i] = np.log(np.dot(w_l_1.T,w_l_1)[0,0])
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
minrmse4 = np.min(rmses3)
argminrmse4 = lambdas[np.argmin(rmses4)]
print "minimum rmse value, ridge regress: ", minrmse4
print "minimum rmse for lamba:", argminrmse4

plt.plot(lambdas,rmses4,'RMSE with regularization paramter(Gradient Descent),\n min RMSE: '
+str(minrmse)+' at '+str(argminrmse))
plt.xlabel('lambda ->')
plt.ylabel('RMSE ->')