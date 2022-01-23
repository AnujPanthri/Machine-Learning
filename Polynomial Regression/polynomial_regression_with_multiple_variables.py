import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

def load_data(str1):
    data=open(str1,'r')
    data=data.read().split('\n')
    data=list(map(lambda x: list(map(float,x.split(','))),data))
    data=np.array(data)
    return data[:,0:2],data[:,2:3]

X,y=load_data("ex1data2.txt")
# X=np.arange(1,20+1,1).reshape(-1,1)
# y=X**2

print(X.shape)
print(y.shape)

class polynomial_regression:
    def __init__(self,X,y,alpha=0.01,order=3,reg=0):
        self.order=order
        self.X=X
        self.num_of_features=X.shape[-1]
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
        # print(self.X[:4,:])
        self.y=y
        self.alpha=alpha
        # self.theta=np.expand_dims([0.7,0.2,1,0.8],axis=1)
        self.theta=np.expand_dims(np.random.rand((self.num_of_features*order)+1),axis=1)
        self.cost_hist=[]
        # print("no. of features :",self.num_of_features)
        # print("scaled_x:",self.scaled_x.shape)
        # print("theta:",self.theta.shape)
        self.reg=reg

    def feature_scaling(self,test_x):
        # print(mu.shape)
        # print(sigma.shape)
        test_x=(test_x-self.mu)/self.sigma
        # print(test_x.shape)
        return test_x

    def polynomial(self,test_x):
        for i in range(2,self.order+1):
            # print(i)
            test_x=np.concatenate([test_x,test_x[:,0:self.num_of_features]**i],axis=-1)
        # print(test_x.shape)
        # print(test_x[:5,:])
        return test_x
    def show(self):
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        
        # Creating plot
        ax.scatter3D(self.X[:,0],self.X[:,1],self.y[:,0] , color = "green")
        # ax.plot3D(self.X[:,1],self.X[:,2],self.y[:,0] , color = "green")

        ax.set_xlabel('Area in square-feet', fontweight ='bold')
        ax.set_ylabel('Room', fontweight ='bold')
        ax.set_zlabel('Price', fontweight ='bold')
        plt.title("Training data")
        plt.show()
    def out(self,test_x):
        # print(test_x.shape)
        # print(self.theta.shape)
        out=np.matmul(test_x,self.theta)
        # print(out.shape)
        return out
    def cost(self):
        m=self.X.shape[0]
        # cost=(1/(2*m))*(np.sum((self.out(self.scaled_x)-self.y)**2))
        cost=(1/(2*m))*(np.sum((self.out(self.scaled_x)-self.y)**2)+(self.reg*np.sum(self.theta[1:,0]**2)) )
        return cost
    def gradient_decent(self):
        m=self.X.shape[0]
        gradients=(1/m)*np.matmul(self.scaled_x.T,(self.out(self.scaled_x)-self.y))
        # self.theta-=self.alpha*gradients
        self.theta[0:1,:]-=self.alpha*gradients[0:1,:]
        self.theta[1:,:]-=self.alpha*(gradients[1:,:]+((self.reg/m)*self.theta[1:,:]))
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
        x_surf, y_surf = np.meshgrid(np.linspace(np.min(self.X[:,0]), np.max(self.X[:,0]), 100),np.linspace(np.min(self.X[:,1]), np.max(self.X[:,1]), 100))
        # x_surf, y_surf = np.meshgrid(np.linspace(np.min(self.X[:,0])-200, np.max(self.X[:,0])+500, 100),np.linspace(np.min(self.X[:,1]), np.max(self.X[:,1]), 100))
        scaled_mesh=self.feature_scaling( self.polynomial( np.concatenate([np.expand_dims(x_surf.ravel(),axis=-1),np.expand_dims(y_surf.ravel(),axis=-1)] , axis=1)))
        # print("scaled_mesh:",scaled_mesh.shape)
        fittedY=self.out(np.concatenate( [np.ones([reduce(lambda a,b: a*b,x_surf.shape),1]),scaled_mesh] , axis=1))
        fittedY=fittedY.reshape(x_surf.shape)
        # print('x_surf:',x_surf.shape)
        # print("y_surf:",y_surf.shape)
        # print("fittedY:",fittedY.shape)

        fig = plt.figure(figsize = (7, 7))
        self.ax = plt.axes(projection ="3d")
        
        # Creating plot
        self.ax.scatter3D(self.X[:,0],self.X[:,1],self.y[:,0] , color = "green")
        # ax.plot3D(self.original_X[:,1],self.original_X[:,2],self.out(self.X)[:,0] , color = "pink") # my own work
        # ax.scatter3D(self.original_X[:,1],self.original_X[:,2],self.out(self.X)[:,0] , color = "red") # my own work
        self.ax.plot_surface(x_surf,y_surf,fittedY, color='green', alpha=0.3)## added later from someone
        self.ax.set_xlabel('Area in square-feet', fontweight ='bold')
        self.ax.set_ylabel('Room', fontweight ='bold')
        self.ax.set_zlabel('Price', fontweight ='bold')
        plt.title("Training data")
        # plt.legend(["Prediction","Training data"])
        plt.show()
        # plt.show(block=False)
        



# pr=polynomial_regression(X,y,order=10)
pr=polynomial_regression(X,y,order=10,reg=100)
pr.show()
# pr.plot_pr()
pr.fit(1000)
pr.plot_training()
pr.plot_pr()



while(True):

    sample_data=np.array(list(map(int,input("Enter Area and No. of Rooms:").split(" "))))
    sample_data=np.expand_dims(sample_data,axis=0)
    sample_scaled_data=pr.polynomial(sample_data) 
    sample_scaled_data=pr.feature_scaling(sample_scaled_data)
    sample_scaled_data=np.concatenate([np.ones([sample_scaled_data.shape[0],1]),sample_scaled_data],axis=-1)
    # print("sample_data",sample_scaled_data.shape)
    # print(test_data)
    test_out=pr.out(sample_scaled_data)[0,0]
    # pr.ax.scatter3D(sample_data[:,0],sample_data[:,1],test_out , color="red")
    # plt.show(block=False)
    print()
    print("Area:",sample_data[0,0],"Rooms:",sample_data[0,1])
    print("Predicted Price:",test_out)