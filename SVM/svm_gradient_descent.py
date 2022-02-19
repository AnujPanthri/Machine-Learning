from matplotlib import pyplot as plt
from kernels import *
import numpy as np
from helper import *
from data_preprocessing import *







class SVM:
    def __init__(self,X,kernel='linear',C=1):
        self.lr=0.3
        self.C=C
        self.kernel=eval(kernel+'_kernel()')
        self.kernel.fit(X)
        self.kernel.num_of_features()
        '''
        in case of gaussian kernel
           input (examples,m)
           theta (m,1)
           out=input*theta=(examples,m)*(m,1)   matrix multiplication
           out (examples,1)
        '''
        self.theta=np.random.randn(1+self.kernel.num_of_features(),1)  # +1 for bias
        # self.theta=np.random.randn(self.kernel.num_of_features(),1)  # +1 for bias
        print('theta:',self.theta.shape)
    def ready_data(self,x):
        x=self.kernel.forward(x)
        x=np.c_[np.ones(x.shape[0]),x]
        return x
    def out(self,x):
        x=self.ready_data(x)
        out=np.matmul(x,self.theta)
        return out
    def loss(self,y_true,y_pred):
        m=y_true.shape[0]
        cost_1=np.maximum(0,1-y_pred) 
        cost_0=np.maximum(0,1+y_pred)
        cost=self.C*np.sum(y_true*cost_1+(1-y_true)*cost_0)+((1/2)*np.sum(self.theta[1:,:]**2))
        # cost=(self.C/m)*np.sum(y_true*cost_1+(1-y_true)*cost_0)+((1/2)*np.sum(self.theta[1:,:]**2))
        return cost
    def gradient_descent(self,test_x,test_y):
        pred=self.out(test_x)
        dc1_pred=(pred<1)*-1
        dc0_pred=(pred>-1)*1
        # dL_pred=(test_y*dc1_pred)+((1-test_y)*dc0_pred) # excluding regularization term      (5,1)
        dL_pred=self.C*((test_y*dc1_pred)+((1-test_y)*dc0_pred)) # excluding regularization term      (5,1)
        # print("dL_pred:",dL_pred.shape)
        dpred_theta=self.ready_data(test_x)  # maybe need to go through the kernel        (5,3)
        
        # print('dpred_theta:',dpred_theta.shape)
        dreg_theta=self.theta[1:,:]                    # (3,1)
        dL_theta=np.zeros_like(self.theta)

        dL_theta[0:1,:]=np.matmul(dpred_theta.T,dL_pred)[0:1,:]
        dL_theta[1:,:]=np.matmul(dpred_theta.T,dL_pred)[1:,:] + dreg_theta

        # dL_theta=np.matmul(dpred_theta.T,dL_pred) + self.theta  # without bias
        # self.theta-=self.lr*dL_theta
        self.theta-=self.lr*dL_theta


    def fit(self,test_x,test_y,iter=100,showat=100):
        
        for i in range(iter):
            self.gradient_descent(test_x,test_y)
            pred=self.out(test_x)
            if (i+1)%showat==0:
                # print(f"epoch {i+1} Loss:",model.loss(test_y,pred))
                print("epoch {:03d} Loss:{:.4f}".format(i+1,model.loss(test_y,pred)))















from scipy.io import loadmat

mat = loadmat("ex6data1.mat")
X = mat["X"]
y = mat["y"]

y=y.reshape(-1,1)

# plt.scatter(X[(y[:,0]==1),0],X[(y[:,0]==1),1],color='green')
# plt.scatter(X[(y[:,0]==0),0],X[(y[:,0]==0),1],color='red')
# plt.show()
# f_s=feature_scaling(X)
# X=f_s.transform(X)

model=SVM(X)
# model=SVM(X,kernel='rbf')

# model.lr=0.0001
# model.lr=0.000005
# model.lr=0.0006
model.lr=0.0001
model.lr=0.000008
# model.C=100
# model.lr=0.0000003
# model.lr=0.000000003
# model.sigma=1
model.sigma=0.1
out=model.out(X)
print("X:",X.shape)
print("Out:",out.shape)
# print("Out:",out)
model.fit(X,y,iter=100000,showat=1000)
show(model,X,y)
# print('kernel',model.kernel.forward(X)[2,4],model.kernel.forward(X)[4,2])