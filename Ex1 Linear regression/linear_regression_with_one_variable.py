import numpy as np
import matplotlib.pyplot as plt

def load_data(str1):
    data=open(str1,'r')
    data=data.read().split('\n')
    data=list(map(lambda x: list(map(float,x.split(','))),data))
    data=np.array(data)
    return data[:,0:1],data[:,1:2]

X,y=load_data("ex1data1.txt")
# print(X.shape)
# print(y.shape)

class linear_regression:
    def __init__(self,X,y,alpha=0.01):
        self.X=np.concatenate([np.ones([X.shape[0],1]),X],axis=1)
        # print(X.shape)
        # print(self.X.shape)
        self.y=y
        self.alpha=alpha
        self.theta=np.expand_dims([0.5,0.8],axis=1)
        self.cost_hist=[]
        # print("theta:",self.theta.shape)
    def show(self):
        plt.figure(figsize=(7,4))
        # plt.plot(X,y,color='orange',marker='o',linestyle='dashed')
        plt.plot(self.X[:,1],self.y,'bo')
        plt.xlabel("X")
        plt.ylabel("Y")
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
        cost=(1/(2*m))*np.sum((self.out(self.X)-self.y)**2)
        return cost
    def gradient_decent(self):
        m=self.X.shape[0]
        gradients=np.matmul(self.X.T,(self.out(self.X)-self.y))
        self.theta-=(self.alpha/m)*(gradients)
    def fit(self,iter=100):
        for i in range(iter):
            self.gradient_decent()
            self.cost_hist.append(self.cost())
            print("cost:",self.cost_hist[-1])
    def plot_training(self):
        epochs=np.arange(1,len(self.cost_hist)+1,1)
        plt.figure(figsize=(7,4))
        plt.plot(epochs,self.cost_hist,'p')
        plt.xlabel("Number of epochs")
        plt.ylabel("Cost(error metric)")
        plt.show()

    def plot_lr(self):
        plt.figure(figsize=(7,4))
        # plt.plot(X,y,color='orange',marker='o',linestyle='dashed')
        plt.plot(self.X[:,1],self.y,'bo')
        plt.plot(self.X[:,1],self.out(self.X),color='orange')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(["Training data","Prediction"])
        plt.show()



lr=linear_regression(X,y)
lr.show()
lr.fit(1000)
lr.plot_training()
lr.plot_lr()