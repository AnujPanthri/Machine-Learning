import numpy as np
import matplotlib.pyplot as plt

def load_data(str1):
    data=open(str1,'r')
    data=data.read().split('\n')
    data=list(map(lambda x: list(map(float,x.split(','))),data))
    data=np.array(data)
    return data[:,0:1],data[:,1:2]

# X,y=load_data("ex1data1.txt")
X=np.arange(1,20+1,1).reshape(-1,1)
y=X**2

# print(X.shape)
# print(y.shape)

class polynomial_regression:
    def __init__(self,X,y,alpha=0.01,order=3):
        self.order=order
        self.X=X
        X=self.polynomial(X)
        self.mu=np.average(X,axis=0)
        self.sigma=np.std(X,axis=0,ddof=1)
        self.mu=np.expand_dims(self.mu,axis=0)
        self.sigma=np.expand_dims(self.sigma,axis=0)
        # print("mu:",self.mu.shape)
        # print("sigma:",self.sigma.shape)

        self.scaled_x=self.feature_scaling(X)
        self.scaled_x=np.concatenate([np.ones([X.shape[0],1]),self.scaled_x],axis=1)
        # print(X.shape)
        # print(self.scaled_x.shape)
        # print(self.X[:4,:])
        self.y=y
        self.alpha=alpha
        # self.theta=np.expand_dims([0.7,0.2,1,0.8],axis=1)
        self.theta=np.expand_dims(np.random.rand(order+1),axis=1)
        self.cost_hist=[]
        # print("theta:",self.theta.shape)

    def feature_scaling(self,test_x):
        # print(mu.shape)
        # print(sigma.shape)
        test_x=(test_x-self.mu)/self.sigma
        # print(test_x.shape)
        return test_x

    def polynomial(self,test_x):
        for i in range(2,self.order+1):
            # print(i)
            test_x=np.concatenate([test_x,test_x[:,0:1]**i],axis=-1)
        # print(test_x.shape)
        # print(test_x[:5,:])
        return test_x
    def show(self):
        fig = plt.figure(figsize = (10, 7))
        plt.plot(self.X[:,0],self.y,"go")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Training data")
        plt.legend(["Training data"])
        plt.show()
    def out(self,test_x):
        # print(test_x.shape)
        # print(self.theta.shape)
        out=np.matmul(test_x,self.theta)
        # print(out.shape)
        return out
    def cost(self):
        m=self.X.shape[0]
        cost=(1/(2*m))*np.sum((self.out(self.scaled_x)-self.y)**2)
        return cost
    def gradient_decent(self):
        m=self.X.shape[0]
        gradients=np.matmul(self.scaled_x.T,(self.out(self.scaled_x)-self.y))
        self.theta-=(self.alpha/m)*(gradients)
    def fit(self,iter=100):
        for i in range(iter):
            self.gradient_decent()
            self.cost_hist.append(self.cost())
            print("cost:",self.cost_hist[-1])
    def plot_training(self):
        epochs=np.arange(1,len(self.cost_hist)+1,1)
        plt.figure(figsize=(7,4))
        plt.plot(epochs,self.cost_hist,color='blue')
        plt.xlabel("Number of epochs")
        plt.ylabel("Cost(error metric)")
        plt.show()

    def plot_pr(self):
        parts=10
        factor=int((np.max(self.X[:,0])-np.min(self.X[:,0]))/parts)
        sample_unscaled_x=np.arange(np.min(self.X[:,0]),np.max(self.X[:,0]),factor).reshape(-1,1)      
        sample_x=self.polynomial(sample_unscaled_x) 
        sample_x=self.feature_scaling(sample_x)
        sample_x=np.concatenate([np.ones([sample_x.shape[0],1]),sample_x],axis=-1)
        sample_y=self.out(sample_x)

        plt.figure(figsize = (10, 7))

        plt.title("Prediction")
        plt.plot(self.X[:,0],y,"go")
        plt.plot(sample_unscaled_x,sample_y,"rx")
        plt.plot(sample_unscaled_x,sample_y,color='orange')
        plt.legend(["Training data","Prediction"])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        



pr=polynomial_regression(X,y,order=3)
pr.show()
pr.plot_pr()
pr.fit(1000)
pr.plot_training()
pr.plot_pr()

# sample_data=np.array([[475]])
# print(sample_data[0,0])
# sample_data=pr.polynomial(sample_data) 
# sample_data=pr.feature_scaling(sample_data)
# sample_data=np.concatenate([np.ones([sample_data.shape[0],1]),sample_data],axis=-1)
# print("sample_data",sample_data.shape)
# print("out:",pr.out(sample_data))

while(True):

    sample_data=np.array([[float(input("Enter number:"))]])
    sample_scaled_data=pr.polynomial(sample_data) 
    sample_scaled_data=pr.feature_scaling(sample_scaled_data)
    sample_scaled_data=np.concatenate([np.ones([sample_scaled_data.shape[0],1]),sample_scaled_data],axis=-1)
    # print("sample_data",sample_scaled_data.shape)
    # print(test_data)
    test_out=pr.out(sample_scaled_data)[0,0]
    print()
    print("Number:",sample_data[0,0])
    print("Prediction:",test_out)