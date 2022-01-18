from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

def load_data(str1):
    data=open(str1,'r')
    data=data.read().split('\n')
    data=list(map(lambda x: list(map(float,x.split(','))),data))
    data=np.array(data)
    return data[:,0:2],data[:,2:]

X,y=load_data("ex1data2.txt")
print(X.shape)
print(y.shape)
class linear_regression:
    def __init__(self,X,y,alpha=0.01):
        self.original_X=np.concatenate([np.ones([X.shape[0],1]),X],axis=1)
        self.mu=np.average(X,axis=0)
        self.sigma=np.std(X,axis=0,ddof=1)

        self.mu=np.expand_dims(self.mu,axis=0)
        self.sigma=np.expand_dims(self.sigma,axis=0)

        X=self.feature_scaling(X)
        self.X=np.concatenate([np.ones([X.shape[0],1]),X],axis=1)
        print("mu:",np.around(self.mu,3))
        print("sigma:",np.around(self.sigma,3))
        # print(X.shape)
        # print(self.X.shape)
        self.y=y
        self.alpha=alpha
        self.theta=np.expand_dims([0.5,1,0.8],axis=1)
        self.cost_hist=[]
        # print("theta:",self.theta.shape)
    def feature_scaling(self,test_x):
        # print(mu.shape)
        # print(sigma.shape)
        test_x=(test_x-self.mu)/self.sigma
        # print(test_x.shape)
        return test_x
    def show(self):
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        
        # Creating plot
        ax.scatter3D(self.original_X[:,1],self.original_X[:,2],self.y[:,0] , color = "green")
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
        x_surf, y_surf = np.meshgrid(np.linspace(np.min(self.original_X[:,1]), np.max(self.original_X[:,1]), 100),np.linspace(np.min(self.original_X[:,2]), np.max(self.original_X[:,2]), 100))
        scaled_mesh=self.feature_scaling(np.concatenate([np.expand_dims(x_surf.ravel(),axis=-1),np.expand_dims(y_surf.ravel(),axis=-1)] , axis=1))
        fittedY=self.out(np.concatenate( [np.ones([reduce(lambda a,b: a*b,x_surf.shape),1]),scaled_mesh] , axis=1))
        fittedY=fittedY.reshape(x_surf.shape)
        print('x_surf:',x_surf.shape)
        print("y_surf:",y_surf.shape)
        print("fittedY:",fittedY.shape)

        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        
        # Creating plot
        ax.scatter3D(self.original_X[:,1],self.original_X[:,2],self.y[:,0] , color = "green")
        # ax.plot3D(self.original_X[:,1],self.original_X[:,2],self.out(self.X)[:,0] , color = "pink") # my own work
        # ax.scatter3D(self.original_X[:,1],self.original_X[:,2],self.out(self.X)[:,0] , color = "red") # my own work
        ax.plot_surface(x_surf,y_surf,fittedY, color='pink', alpha=0.3)## added later from someone
        ax.set_xlabel('Area in square-feet', fontweight ='bold')
        ax.set_ylabel('Room', fontweight ='bold')
        ax.set_zlabel('Price', fontweight ='bold')
        plt.title("Training data")
        # plt.legend(["Prediction","Training data"])
        plt.show()
        



lr=linear_regression(X,y)
lr.show()
lr.plot_lr()
lr.fit(1000)
# lr.plot_training()
lr.plot_lr()



# test_data=np.array([[1650,3]])
# test_data=np.array([[1650,3],[1300,4]])
# scaled_data=lr.feature_scaling(test_data)
# test_out=lr.out( np.concatenate( [np.ones([test_data.shape[0],1]),scaled_data] ,axis=-1) )

# for i in range(test_data.shape[0]):
#     print()
#     print("Area:",test_data[i,0],"Rooms:",test_data[i,1])
#     print("Predicted Price:",test_out[i,0])

while(True):

    test_data=np.array(list(map(int,input("Enter Area and No. of Rooms:").split(" "))))
    test_data=np.expand_dims(test_data,axis=0)
    # print(test_data)
    scaled_data=lr.feature_scaling(test_data)
    test_out=lr.out( np.concatenate( [np.ones([test_data.shape[0],1]),scaled_data] ,axis=-1) )
    print()
    print("Area:",test_data[0,0],"Rooms:",test_data[0,1])
    print("Predicted Price:",test_out[0,0])