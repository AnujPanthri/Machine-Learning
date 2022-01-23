import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('linpts.txt')
X,y=data[:,0:2],data[:,2:]
print('X:',X.shape)
print('y:',y.shape)

class logistic_regression:
    def __init__(self,X,y,alpha=0.01,reg=0):
        self.original_x=X
        self.mu=np.mean(X,axis=0).reshape(1,-1)
        self.sigma=np.std(X,axis=0,ddof=1).reshape(1,-1)
        # print("mu:",self.mu.shape)
        # print("sigma:",self.sigma.shape)
        # self.X=X
        self.X=self.feature_scaling(X)
        self.X=np.concatenate([ np.ones([X.shape[0],1]),self.X],axis=-1)
        print(self.X.shape)
        self.y=y
        self.alpha=alpha
        self.theta=np.random.rand(X.shape[1]+1).reshape(-1,1)
        print(self.theta.shape)
        self.cost_hist=[]
        self.reg=reg

    def ready_data(self,test_x):
        test_x=self.feature_scaling(test_x)
        test_x=np.c_[np.ones(test_x.shape[0]),test_x]
        return test_x

    def feature_scaling(self,test_x):
        test_x=(test_x-self.mu)/self.sigma
        # print("test_x_after_feature_scaling:",test_x.shape)
        return test_x
    def scaled_to_normal(self,test_x):
        test_x=(test_x*self.sigma)+self.mu
        return test_x
    def show(self):
        plt.figure(figsize=(10,7))
        # plt.plot(self.original_x[:,0],self.original_x[:,1],"go")
        plt.plot(self.original_x[(self.y==0)[:,0],0],self.original_x[(self.y==0)[:,0],1],"go")
        plt.plot(self.original_x[(self.y==1)[:,0],0],self.original_x[(self.y==1)[:,0],1],"bo")
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend(["training data"])
        plt.show()
    def sigmoid(self,test_x):
        z=1/(1+np.exp(-test_x))
        return z
    def out(self,test_x):
        out=self.sigmoid(np.matmul(test_x,self.theta))
        return out
    def cost(self,test_x):
        m=test_x.shape[0]
        epsilon = 1e-5 
        # cost=-(1/m)* np.sum((y*np.log(self.out(test_x)) ) + (1-y)*np.log(1-self.out(test_x)) )
        cost=-(1/m)* np.sum((self.y*np.log(self.out(test_x)+epsilon) ) + (1-self.y)*np.log(1-self.out(test_x)+epsilon) )+((self.reg/(2*m))*np.sum(self.theta[1:,0]**2))
        return cost
    def gradient_decent(self):
        m=self.X.shape[0]
        # print(self.theta.shape)
        grads=(1/m)*np.matmul( self.X.T ,(self.out(self.X)-self.y))
        #self.theta-=self.alpha*grads  # without regularization
        self.theta[0:1,:]-=self.alpha*grads[0:1,:] 
        self.theta[1:,:]-=self.alpha*(grads[1:,:]+((self.reg*self.theta[1:,:])/m))
    def fit(self,iter=1000):
        for i in range(iter):
            self.gradient_decent()
            self.cost_hist.append(self.cost(self.X))
            print(self.cost_hist[-1])
    def plot_training(self):
        plt.figure(figsize=(10,7))
        plt.plot(np.arange(1,len(self.cost_hist)+1,1),self.cost_hist,"b")
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.show()
    def acc(self,test_x=0,test_y=0,show=True):
        if isinstance(test_x,np.ndarray):
            y_true=test_y
            y_pred=(self.out(self.ready_data(test_x))>=0.5)*1
        else:
            y_true=self.y
            y_pred=(self.out(self.X)>=0.5)*1
        # print(np.unique(y_pred))
        total=np.sum(np.ones_like(y_true))
        match=np.sum((y_true==y_pred)*1)
        acc=(match/total)*100
        if show:
            print("acc:","{:.2f}".format(acc))
        return acc
    def plot_lr(self):

        x_min, x_max = self.original_x[:, 0].min() - 0.5, self.original_x[:, 0].max() + 0.5
        y_min, y_max = self.original_x[:, 1].min() - 0.5, self.original_x[:, 1].max() + 0.5
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # print("see",xx.shape)
        data_to_input=self.ready_data(np.c_[xx.ravel(), yy.ravel()])
        zz=self.out(data_to_input)
        zz=(zz>=0.5)*1
        zz=zz.reshape(xx.shape)

        xx,yy=xx.reshape(zz.shape),yy.reshape(zz.shape)
        plt.figure(figsize=(10,7))
        plt.title(f"Prediction(decision boundary) acc:{self.acc(show=False)}")
        plt.plot(self.original_x[(self.y==0)[:,0],0],self.original_x[(self.y==0)[:,0],1],"go")
        plt.plot(self.original_x[(self.y==1)[:,0],0],self.original_x[(self.y==1)[:,0],1],"bo")
        
        plt.contourf(xx, yy, zz, cmap='Paired')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()

lr=logistic_regression(X,y)
# lr=logistic_regression(X,y,reg=10)
lr.show()
# lr.plot_lr()
lr.fit(1000)
# lr.plot_training()
lr.plot_lr()
lr.acc()