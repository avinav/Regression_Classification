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
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD 

    return np.dot(np.dot(linalg.inv(np.dot(X.T, X)), X.T), y)
    
def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1 (? N x 1)
    # Output:
    # rmse
    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    return np.sqrt(np.sum(np.square(ytest - np.dot(Xtest, w))))/N


#--------------------Script starts here-------------------------
plt.close('all')

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
print "||W|| without intercept: ", np.dot(w.T,w)[0,0]
# Show predicted data
fig = plt.figure()
y_pred = np.dot(Xtest,w)
ax = plt.gca()
ax.grid(True,linestyle='-',color='0.75')
ax.set_xlim([0, 200])
ax.set_xticks(np.arange(0,200,10))
plt.scatter(range(Xtest.shape[0]),y_pred, label='predicted label',marker='.')
plt.scatter(range(Xtest.shape[0]),ytest, label='true label',color='r',marker='.')
for i in range(0,Xtest.shape[0]):
    plt.plot([i, i], [y_pred[i],ytest[i]],color='black')
fig.suptitle('Predicted and true labels of Diabetes test data without intercept, RMSE: ' +str(mle))
plt.xlabel('Data points ->')
plt.ylabel('Diabetes intensity ->')
plt.legend()
fig.show()
savefig("results/ole_no_intercept.png", format="png")

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
y_pred = np.dot(Xtest_i,w_i)
print "||W|| with intercept: ", np.dot(w_i.T,w_i)[0,0]
# Show predicted data
fig = plt.figure()
ax = plt.gca()
ax.grid(True,linestyle='-',color='0.75')
ax.set_xlim([0, 200])
ax.set_xticks(np.arange(0,200,10))
plt.scatter(range(Xtest_i.shape[0]),y_pred, label='predicted label',marker='.')
plt.scatter(range(Xtest_i.shape[0]),ytest, label='true label',color='r',marker='.')
for i in range(0,Xtest_i.shape[0]):
    plt.plot([i, i], [y_pred[i],ytest[i]],color='black')
fig.suptitle('Predicted and true labels of Diabetes test data (with intercept), RMSE: ' +str(mle_i))
plt.xlabel('Data points ->')
plt.ylabel('Diabetes intensity ->')
plt.legend()
fig.show()
savefig("results/ole_with_intercept.png", format="png")

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

