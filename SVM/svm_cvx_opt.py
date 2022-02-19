from matplotlib import pyplot as plt
from kernels import *
import numpy as np
from helper import *
from data_preprocessing import *







class svm:       # large margin classifier
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
        # self.theta=np.random.randn(1+self.kernel.num_of_features(),1)  
        # self.theta=np.zeros((1+self.kernel.num_of_features(),1))  

        self.w=np.zeros((self.kernel.num_of_features(),1))  
        self.b=np.zeros((1,1)) 
        
        # print('w:',self.w.shape)
        # print('b:',self.b.shape)
        # print('theta:',self.theta.shape)
    def out(self,x):
        x=self.kernel.forward(x)
        out=np.matmul(x,self.w) + self.b
        return out

    # def loss(self,y_true,y_pred):
    #     # hinge loss
    #     m=y_true.shape[0]
    #     '''
    #     cost_1=0 y_true=1 if y_pred>=1 #cost_1=somenumber if y_pred<1

    #     cost_0=0 y_true=-1 if y_pred<=-1 #cost_0=somenumber if y_pred>-1

    #     cost=y(wx+b)>=1
    #     '''
    #     y_true=np.where(y_true>0,1,-1)
    #     c=np.maximum(0,1-(y_true*y_pred))   # 
    #     cost=self.C*np.sum(c)
    #     # cost=(self.C*np.sum(c))/m
    #     # print("classification error:",cost)
        
    #     cost+=((1/2)*np.sum(self.w**2))
    #     # cost+=((1/2)*np.sum(self.theta[1:,:]**2))
    #     return cost

    # def gradient_descent(self,test_x,test_y):
    #     test_y=np.where(test_y>0,1,-1)
    #     pred=self.out(test_x)
    #     #pred<=0 dc_pred=0
    #     #pred>0 dc_pred=-y_true
    #     # dc_pred=(pred>0)*-test_y
    #     dc_pred=np.where(1-(test_y*pred)>0,-test_y,0) # right 
    #     dL_pred=self.C*dc_pred # excluding regularization term      (5,1)
    #     dpred_theta=self.kernel.forward(test_x)  # maybe need to go through the kernel        (5,3)
    #     dreg_theta=self.theta[1:,:]                    # (3,1)
    #     dL_theta=np.zeros_like(self.theta)

    #     dL_theta[0:1,:]=np.matmul(dpred_theta.T,dL_pred)[0:1,:]
    #     dL_theta[1:,:]=np.matmul(dpred_theta.T,dL_pred)[1:,:] + dreg_theta

    #     self.theta-=self.lr*dL_theta
    # def lr_decay(self):
    #     self.lr-=self.lr-0.001

    # def fit(self,test_x,test_y,iter=100,showat=100):
        
    #     for i in range(iter):
    #         self.gradient_descent(test_x,test_y)
    #         # self.lr_decay()
    #         pred=self.out(test_x)
    #         if (i+1)%showat==0:
    #             # print(f"epoch {i+1} Loss:",model.loss(test_y,pred))
    #             print("epoch {:03d} Loss:{:.4f}".format(i+1,model.loss(test_y,pred)))
    
    
    def qp_solver(self,X,y):
        from cvxopt import matrix, solvers
        
        y=np.where(y!=1,-1,1)
        # y=np.where(y!=1,1,-1)
        X,y=X.astype("float64"),y.astype('float64')
        m, n = X.shape
        K=self.kernel.forward(X,X)
        print(np.unique(y))

        # main equation to solve
        P = matrix(np.matmul(y,y.T) * K)
        q = matrix(np.ones((m, 1)) * -1)

        # for less than equal to constraint
        G = matrix(np.vstack((np.eye(m) * -1, np.eye(m))))        
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))

        # for equality constraint
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))    
        
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x'])
        ind = (alphas > 1e-4).flatten()
        self.sv = X[ind]
        self.sv_y = y[ind]
        self.alphas = alphas[ind]

        b = self.sv_y - np.sum(self.kernel.forward(self.sv, self.sv) * self.alphas * self.sv_y, axis=0)
        self.b = np.sum(b) / b.size
        
        self.w=self.alphas * self.sv_y      
        self.kernel.fit(self.sv)
        print('number of support vectors found:',self.sv.shape[0])
            

    def predict(self, X):
        """
        Predict labels for input data X

        :param numpy.array X : Input data
        :return np.array : Predictions
        """
        # K=self.kernel.forward(X,self.sv)
        K=self.kernel.forward(X)
        

        prod=np.matmul(K,self.w)+self.b
        y=prod
        y = y.reshape(-1, 1)
        y = (y>0)*1
        
        return y

















from scipy.io import loadmat
mat = loadmat("ex6data1.mat")
X = mat["X"]
y = mat["y"]




y=y.reshape(-1,1)

print(np.unique(y))
print(X.shape)
print(y.shape)

model=svm(X)
# model=svm(X,kernel='rbf')

model.C=100
model.kernel.sigma=0.1

model.qp_solver(X,y)
model.predict(X)
show(model,X,y,0)
