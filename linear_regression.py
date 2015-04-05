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
    return np.sqrt(np.mean(np.square(ytest - np.dot(Xtest, w))))


#--------------------Script starts here-------------------------
plt.close('all')

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

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
fig.suptitle('Predicted and true labels of Diabetes test data without intercept')
plt.xlabel('Data points ->')
plt.ylabel('Diabetes intensity ->')
plt.legend()
fig.show()
#savefig("results/ole_no_intercept.png", format="png")

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
y_pred = np.dot(Xtest_i,w_i)
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
fig.suptitle('Predicted and true labels of Diabetes test data (with intercept)')
plt.xlabel('Data points ->')
plt.ylabel('Diabetes intensity ->')
plt.legend()
#fig.show()
savefig("results/ole_with_intercept.png", format="png")

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

'''
# Low rank approximation of diabetes train data
def low_rank_approx(X,r):
    U, s, Vh = linalg.svd(X)
    s [r:] = 0
    sk = linalg.diagsvd(s, U.shape[1], Vh.shape[0])
    X_app = np.dot(U, np.dot(sk, Vh))
    X_app = X_app[:,:r]
    return X_app

X_app = low_rank_approx(X,2)
Xtest_app = low_rank_approx(Xtest,2)
w = learnOLERegression(X_app,y)

xx, yy = np.meshgrid(np.linspace(np.min(Xtest_app[:,0]),np.max(Xtest_app[:,0]),50),
np.linspace(np.min(Xtest_app[:,1]),np.max(Xtest_app[:,1]),50))
grid = np.vstack((xx.reshape(xx.size),yy.reshape(yy.size))).T

y_grid = np.dot(grid,w)
y_pred = np.dot(Xtest_app,w)
y_mesh = y_grid.reshape(xx.shape[0],yy.shape[0])
#colors = [cm.rainbow(np.linspace(0,1,np.unique(y_dt_train).size))]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(grid[:,0],grid[:,1],zs=y_grid)
ax.plot_wireframe(xx,yy,y_mesh,label='ole plane')
ax.scatter(Xtest_app[:,0],Xtest_app[:,1],zs=y_pred,color='red',s=40,label='predicted test values')
ax.scatter(Xtest_app[:,0],Xtest_app[:,1],zs=ytest,color='green',s=40,label='true test values')
fig.suptitle('Diabetes data in 2-D with PCA and corresponding plane learnt(without intercept)')
ax.set_xlabel("reduced dim-1")
ax.set_ylabel("reduced dim-2")
ax.set_zlabel("diabetes intensity")
plt.legend()
fig.show()

'''