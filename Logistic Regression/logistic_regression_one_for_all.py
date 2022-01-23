import numpy as np
import matplotlib.pyplot as plt


from sklearn import datasets

# X,y = datasets.make_moons(n_samples=1000, 
#                           shuffle=True, 
#                           noise=0.09, 
#                           random_state=4)

X,y = datasets.make_blobs(n_samples=1000,n_features=2,centers=4)
# X,y = datasets.make_circles(n_samples=100, noise=0.03,)
y = np.reshape(y,(len(y),1))
print('X:',X.shape)
print('y:',y.shape)
print("y values:",np.unique(y))

class logistic_regression:
    def __init__(self,X,y,alpha=0.01,order=1,reg=0):
        self.order=order
        self.original_x=X
        self.num_of_features=X.shape[-1]
        
        self.X=self.polynomial(X)
        self.mu=np.mean(self.X,axis=0).reshape(1,-1)
        self.sigma=np.std(self.X,axis=0,ddof=1).reshape(1,-1)
        # print("mu:",self.mu.shape)
        # print("sigma:",self.sigma.shape)
        self.X=self.feature_scaling(self.X)
        self.X=np.concatenate([ np.ones([X.shape[0],1]),self.X],axis=-1)
        # print("X:",self.X.shape)
        self.y=y
        self.alpha=alpha
        self.theta=np.random.rand((X.shape[1]*self.order)+1).reshape(-1,1)
        # print("theta:",self.theta.shape)
        self.cost_hist=[]
        self.reg=reg
    def ready_data(self,test_x):
        test_x=self.polynomial(test_x)
        test_x=self.feature_scaling(test_x)
        test_x=np.c_[np.ones(test_x.shape[0]),test_x]
        return test_x


    def polynomial(self,test_x):
        for i in range(2,self.order+1):
            test_x=np.concatenate([test_x,test_x[:,:self.num_of_features]**i],axis=-1)
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
        # plt.plot(self.original_x[(self.y==2)[:,0],0],self.original_x[(self.y==2)[:,0],1],"ro")
        plt.xlabel('X')
        plt.ylabel('y')
        # plt.legend(["training data"])
        plt.title("training data")
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
        grads=(1/m)*np.matmul( self.X.T ,(self.out(self.X)-self.y))
        # self.theta-=self.alpha*grads
        self.theta[0:1,:]-=self.alpha*grads[0:1,:]
        self.theta[1:,:]-=self.alpha*(grads[1:,:]+((self.reg*self.theta[1:,:])/m))
    def fit(self,iter=1000,show=False):
        for i in range(iter):
            self.gradient_decent()
            self.cost_hist.append(self.cost(self.X))
            if show:
                print("{:.2f}".format(self.cost_hist[-1]) )
    def plot_training(self):
        plt.figure(figsize=(10,7))
        plt.plot(np.arange(1,len(self.cost_hist)+1,1),self.cost_hist,"b")
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.show()
    def acc(self):
        y_true=self.y
        y_pred=(self.out(self.X)>=0.5)*1
        total=np.sum(np.ones_like(y_true))
        match=np.sum((y_true==y_pred)*1)
        acc=(match/total)*100
        print("acc:","{:.2f}".format(acc))
    def plot_lr(self):

        x_min, x_max = self.original_x[:, 0].min() - 0.5, self.original_x[:, 0].max() + 0.5
        y_min, y_max = self.original_x[:, 1].min() - 0.5, self.original_x[:, 1].max() + 0.5
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # print("see",xx.shape)
        data_as_input=np.c_[xx.ravel(), yy.ravel()]
        data_as_input=np.c_[np.ones(xx.ravel().shape[0]),self.feature_scaling(self.polynomial(data_as_input))]
        # print('data_as_input',data_as_input.shape)
        zz=self.out(data_as_input)
        zz=(zz>=0.5)*1
        zz=zz.reshape(xx.shape)

        # normal_data=self.scaled_to_normal(np.c_[xx.ravel(), yy.ravel()])
        # xx,yy=normal_data[:,0].reshape(zz.shape),normal_data[:,1].reshape(zz.shape)
        plt.figure(figsize=(10,7))
        plt.title("Prediction(decision boundary)")
        plt.plot(self.original_x[(self.y==0)[:,0],0],self.original_x[(self.y==0)[:,0],1],"go")
        plt.plot(self.original_x[(self.y==1)[:,0],0],self.original_x[(self.y==1)[:,0],1],"bo")
        
        plt.contourf(xx, yy, zz, cmap='Paired')
        
        plt.xlabel('X')
        plt.ylabel('y')
        # plt.legend(["training data"])
        plt.show()



class one_for_all_logistic_regression:
    def __init__(self,X,y):
        self.all_models=[]
        self.X=X
        self.y=y
        self.classes=np.unique(self.y)

    def train_all(self,iter=2000,alpha=0.1,order=3,reg=0,show=False):
        for i in self.classes:
            sub_x=self.X
            sub_y=np.zeros_like(self.y)
            sub_y[(self.y==i)[:,0],:]=1
            lr=logistic_regression(sub_x,sub_y,alpha=alpha,order=order,reg=reg)
            # if show:
                # lr.show()            
            lr.fit(iter)
            if show:
                lr.plot_training()
                # lr.acc()
                # lr.plot_lr()
            self.all_models.append(lr)
    def get_results(self,test_x):
        
        for i in self.classes:
            if i==0:
                out=self.all_models[i].out(test_x)
            else:
                out=np.c_[out,self.all_models[i].out(test_x)]
        out=np.argmax(out,axis=-1).reshape(-1,1)
        # print("out:",out.shape)
        return out
    def acc(self):
        y_true=self.y
        y_pred=self.get_results(self.all_models[0].ready_data(self.X))
        total=np.sum(np.ones_like(y_true))
        match=np.sum((y_true==y_pred)*1)
        acc=(match/total)*100
        for i in self.classes:
            print((i+1)," model:",sep='',end='')
            self.all_models[i].acc()
        print("whole model's acc:","{:.2f}".format(acc))
    def decision_boundary(self):

        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # print("see",xx.shape)
        data_as_input=np.c_[xx.ravel(), yy.ravel()]
        data_as_input=self.all_models[0].ready_data(data_as_input)
        # print('data_as_input',data_as_input.shape)
        zz=self.get_results(data_as_input)
        # zz=(zz>=0.5)*1
        zz=zz.reshape(xx.shape)

        # normal_data=self.scaled_to_normal(np.c_[xx.ravel(), yy.ravel()])
        # xx,yy=normal_data[:,0].reshape(zz.shape),normal_data[:,1].reshape(zz.shape)
        plt.figure(figsize=(10,7))
        plt.title("Prediction(decision boundary)")
        colors=['go','bo','ro','co','mo','yo','ko','wo']
        for i in self.classes:
            plt.plot(self.X[(self.y==i)[:,0],0],self.X[(self.y==i)[:,0],1],colors[i])
        
        plt.contourf(xx, yy, zz, cmap='Paired')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()
        
        
model=one_for_all_logistic_regression(X,y)
# model.train_all(show=True)
model.train_all(iter=2000,alpha=0.1,order=3,reg=0,show=False)
processed_X=model.all_models[0].ready_data(X)
model.get_results(processed_X)
model.acc()
model.decision_boundary()
# print("dfdsfsd")