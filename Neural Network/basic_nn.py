import numpy as np

X=np.array([[0,0],
          [0,1],
          [1,0],
          [1,1],
         ])
y=np.array([[0],
            [1],
            [1],
            [1],
            ])

class nn:
    def __init__(self,X,y):
        self.X=X.T
        self.y=y.T
        print("self.X:",self.X.shape)
        print("self.y:",self.y.shape)
        a1=self.X
        self.a1=np.r_[np.ones([1,self.X.shape[-1]]),a1]
        self.theta1=np.random.rand(1*self.X.shape[0]+1).reshape([1,self.X.shape[0]+1])
        print('theta1:',self.theta1.shape)
        print("a1:",self.a1.shape)
        # print(a1)
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def forward(self):
        self.z2=self.theta1@self.a1
        self.a2=self.sigmoid(self.z2)
    def loss(self):
        m=self.y.shape[-1]
        cost=-(1/m)*np.sum(self.y*np.log(self.a2)+(1-self.y)*np.log(1-self.a2))
        return cost
    def backward(self):
        m=self.y.shape[-1]
        self.dz2=self.a2-self.y
        self.dtheta1=self.dz2@self.a1.T
        self.theta1-=(1/m)*self.dtheta1
        # print("dtheta1:",self.dtheta1.shape)


model=nn(X,y)
# model.theta1=np.array([[-30,20,20]])
for i in range(1000):
    model.forward()
    model.backward()
    print("loss:",model.loss())

print('final theta:',model.theta1)
print('final predicts:',model.a2.T)
print('y_true',y)