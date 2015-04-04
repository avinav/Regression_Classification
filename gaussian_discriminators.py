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
def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    N = X.shape[0]
    d = X.shape[1]
    # initialization
    y = y.reshape(y.size)
    classes = np.unique(y)
    means = np.zeros((d,classes.size))
    
    for cl in range(classes.size):
        means[:,cl] = np.mean(X[y==classes[cl]],0) 
    
    return means, np.cov(X.T)   
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
    N = X.shape[0]
    d = X.shape[1]
    # initialization
    y = y.reshape(y.size)
    classes = np.unique(y)
    means = np.zeros((d,classes.size))
    covmats = [np.zeros((d,d))] * classes.size
    
    for cl in range(classes.size):
        means[:,cl] = np.mean(X[y==classes[cl]],0)
        covmats[cl] = np.cov(X[y==classes[cl]].T)

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    invcovmat = linalg.inv(covmat)
    covmatdet = linalg.det(covmat)
    ydist = np.zeros((Xtest.shape[0],means.shape[1]))
    for i in range(means.shape[1]):
        ydist[:,i] = np.exp(-0.5*np.sum((Xtest - means[:,i])* 
        np.dot(invcovmat, (Xtest - means[:,i]).T).T,1))/(np.sqrt(np.pi*2)*(covmatdet**2))
    ylabel = np.argmax(ydist,1)
    ylabel = ylabel + 1
    ytest = ytest.reshape(ytest.size)
    acc = 100*np.mean(ylabel == ytest)
    return acc, ylabel

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD

    ydist = np.zeros((Xtest.shape[0],means.shape[1]))
    for i in range(means.shape[1]):
        invcovmat = linalg.inv(covmats[i])
        covmatdet = linalg.det(covmats[i])
        ydist[:,i] = np.exp(-0.5*np.sum((Xtest - means[:,i])* 
        np.dot(invcovmat, (Xtest - means[:,i]).T).T,1))/(np.sqrt(np.pi*2)*(covmatdet**2))
    ylabel = np.argmax(ydist,1)
    ylabel = ylabel + 1
    ytest = ytest.reshape(ytest.size)
    acc = 100*np.mean(ylabel == ytest)
    return acc, ylabel   
    #return acc

def plot_data(data0, label0, title, subtitle, *args):
    
    colors = iter(cm.rainbow(np.linspace(0,1,5)))
    plt.clf()
    fig = plt.figure()
    ax = plt.gca()
    if (len(args) == 2):
        means, covmats = args
    elif (len(args) == 4):
        means, covmats, traindata, trainlabel = args
        trainlabel = trainlabel.reshape(trainlabel.size)
    for i in range(1,6):
        cl = next(colors)
        plt.scatter(data0[(label0==i).reshape((label0==i).size),0],
        data0[(label0==i).reshape((label0==i).size),1],color=cl)
        # Draw ellipse with 1 standard deviation
        if (len(args) >= 2):
            # Compute eigen vectors and value for covmat to plot an ellipse
            lambda_, v = np.linalg.eig(covmats[i-1])
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=(means[0,i-1], means[1,i-1]),
                        width=lambda_[0]*2, height=lambda_[1]*2,
                        angle=np.rad2deg(np.arccos(v[0, 0])))
            ell.set_facecolor('none')
            ax.add_artist(ell)
            #Add mean points
            plt.scatter(means[0,i-1], means[1,i-1],c='black',s=30);
            if (len(args) == 4):
                plt.scatter(traindata[trainlabel==i,0],
                traindata[trainlabel==i,1],color=cl,
                edgecolor='black')
    fig.suptitle(title)
    ax.set_title(subtitle)
    fig.show()
    return
            
#------------------Script Starts Here------------------------------
# Close figures
plt.close("all")

# Load sample data
data = pickle.load(open("sample.pickle","rb"))
[data0, label0, test0, label1] = data

# LDA Learn means and covariance
means, covmat = ldaLearn(data0,label0)
acc, ylabel = ldaTest(means,covmat,test0,label1)
print ('LDA Accuracy: ' + str(acc) + '%')
# Create meshgrid
xx, yy = np.meshgrid(np.linspace(0,16,100), np.linspace(0,16,100))
grid = np.vstack((xx.reshape(xx.size),yy.reshape(yy.size))).T

acctemp, gridlabel = ldaTest(means,covmat, grid, label1)
plot_data(grid,gridlabel,'LDA Accuracy: ' + str(acc) +'%', 
'black dots/solid-dots/ellipse -> training data points/mean/covariance', means,[covmat]*means.shape[1],
data0,label0)

# Plot sample training data with LDA data
#plot_data(data0,label0,'(LDA) Sample data 1 with class mean and common covariance(1 SD)',
#means,[covmat]*means.shape[1])
#plot_data(data1,label1,'Sample data 2')        

# QDA Learn mean and covariances
qda_means, qda_covmats = qdaLearn(data0, label0)
acc, ylabel = qdaTest(qda_means,qda_covmats,test0,label1)
print ('QDA Accuracy: ' + str(acc) + '%')

acctemp, gridlabel = qdaTest(qda_means,qda_covmats, grid, label1)
plot_data(grid,gridlabel,'QDA Accuracy: ' + str(acc) +'%',
'black dots/solid-dots/ellipse -> training data points/mean/covariance', qda_means,qda_covmats,
data0,label0)

# Plot sample training data with QDA data
# plot_data(data0,label0,'(QDA) Sample data 1 with class mean and covariances(1 SD)',
# qda_means,qda_covmats)

# Load diabetes data
dtdata = pickle.load(open('diabetes.pickle','rb'))
[x_dt_train, y_dt_train, x_dt_test, y_dt_test] = dtdata

# Low rank approximation of diabetes train data
U, s, Vh = linalg.svd(x_dt_train.T)
temps = np.copy(s)
s [2:] = 0
sk = linalg.diagsvd(s, U.shape[1], Vh.shape[0])
dt0_app = np.dot(U, np.dot(sk, Vh))
dt0_app = dt0_app[:,:2]

#colors = [cm.rainbow(np.linspace(0,1,np.unique(y_dt_train).size))]
#fig = plt.figure()
#plt.scatter(dt0_app[:,0],dt0_app[:,1])
#fig.suptitle('Diabetes data in 2-D with low rank approximation')
#fig.show()

