import numpy as np
from linear_regression import learnOLERegression,testOLERegression
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import pickle
from ridge_regression import learnRidgeRegression

# Low rank approximation of diabetes train data
def low_rank_approx(X,r):
    U, s, Vh = linalg.svd(X)
    s [r:] = 0
    sk = linalg.diagsvd(s, U.shape[1], Vh.shape[0])
    X_app = np.dot(U, np.dot(sk, Vh))
    X_app = X_app[:,:r]
    return X_app

#----------------Script starts here---------------------------
plt.close('all')

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

X_app = low_rank_approx(X,10)
Xtest_app = low_rank_approx(Xtest,10)

#plt.scatter(X_app,y,c='b',label='train')
#plt.scatter(Xtest_app,ytest,c='r',label='test')
#plt.legend()
#plt.show()

# add intercept
X_app = np.concatenate((np.ones((X_app.shape[0],1)),X_app),axis=1)
Xtest_app = np.concatenate((np.ones((Xtest_app.shape[0],1)),Xtest_app),axis=1)
w = learnOLERegression(X_app,y)
y_pred = np.dot(Xtest_app,w)
rmse = testOLERegression(w,Xtest_app,ytest)
print "rmse value: ",rmse

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
fig.suptitle('Predicted and true labels test data (11-D with intercept), RMSE: ' +str(rmse))
plt.xlabel('Data points ->')
plt.ylabel('Diabetes intensity ->')
plt.legend()
fig.show()

#Ridge Regression
k = 200
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
w_mag = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_app,y,lambd)
    w_mag[i] = np.log(np.dot(w_l.T,w_l)[0,0])
    rmses3[i] = testOLERegression(w_l,Xtest_app,ytest)
    i = i + 1
minrmse = np.min(rmses3)
argminrmse = lambdas[np.argmin(rmses3)]
print "minimum rmse value, ridge regress: ", minrmse
print "minimum rmse for lamba:", argminrmse
fig = plt.figure()
plt.plot(lambdas,rmses3)
fig.suptitle('Variation of RMSE(11-D) with regularization paramter,\n min RMSE: '
+str(minrmse)+' at '+str(argminrmse))
plt.xlabel('lambda ->')
plt.ylabel('RMSE ->')
fig.show()
fig = plt.figure()
plt.plot(lambdas,w_mag)
fig.suptitle('Variation of ln(||W||) (11-D) with regularization paramter')
plt.xlabel('lambda ->')
plt.ylabel('ln(||W||) ->')
fig.show()

## Plot 3-d reduced
'''
xx, yy = np.meshgrid(np.linspace(np.min(Xtest_app[:,1]),np.max(Xtest_app[:,1]),50),
np.linspace(np.min(Xtest_app[:,2]),np.max(Xtest_app[:,2]),50))
grid = np.vstack((xx.reshape(xx.size),yy.reshape(yy.size))).T

## add intercept in grid
grid = np.concatenate((np.ones((grid.shape[0],1)), grid), axis=1)

y_grid = np.dot(grid,w)
y_mesh = y_grid.reshape(xx.shape[0],yy.shape[0])

#colors = [cm.rainbow(np.linspace(0,1,np.unique(y_dt_train).size))]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(grid[:,0],grid[:,1],zs=y_grid)
ax.plot_wireframe(xx,yy,y_mesh,label='ole plane')
ax.scatter(Xtest_app[:,1],Xtest_app[:,2],zs=y_pred,color='red',s=40,label='predicted test values')
ax.scatter(Xtest_app[:,1],Xtest_app[:,2],zs=ytest,color='green',s=40,label='true test values')
fig.suptitle('Diabetes data(with intercept added) in 2-D (with intercept as 3rd D) , RMSE: ' +str(rmse))
ax.set_xlabel("reduced dim-1")
ax.set_ylabel("reduced dim-2")
ax.set_zlabel("diabetes intensity")
plt.legend()
fig.show()
'''