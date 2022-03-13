import numpy as np


ratings=np.loadtxt('ratings.txt')
item_features=np.loadtxt('item_features.txt')

items=[f"item {i+1}" for i in range(ratings.shape[0])]
users=[f"user {i+1}" for i in range(ratings.shape[1])]
print(ratings)
print("item_features:")
print(item_features)
r=(ratings!=-1)*1
print(r)
# print(items)
# print(users)


class content_based:
    '''
        X (5,2)             excluding bias term
        theta (2,4)
        y (5,4)
    '''
    def __init__(self,mean_normalization=False,use_bias=True):
        self.use_bias=use_bias
        self.mean_normalization=mean_normalization
        self.lr=0.01
        self.reg=0
    def predict(self,X,backend=False):
        if not backend:
            if self.use_bias: 
                X=np.c_[np.ones((X.shape[0],1)),X]
        h_x=np.matmul(X,self.theta)
        if not backend and self.mean_normalization:
                h_x=h_x+self.avg_items
                # h_x=h_x+self.avg_users
        return h_x
    def loss(self,X,y,r,backend=False):
        if not backend:
            if self.mean_normalization:
                y=y-self.avg_items
                # y=y-self.avg_users
            if self.use_bias: 
                X=np.c_[np.ones((X.shape[0],1)),X]
        if self.use_bias:
            reg_term=(self.reg/2)*np.sum(self.theta[:-1,:]**2)
        else:
            reg_term=(self.reg/2)*np.sum(self.theta**2)
        h_x=self.predict(X,backend=True)
        # loss=(1/2)*np.sum(((h_x-y)**2)*r) + reg_term
        loss=(1/2)*np.sum(((h_x-y)*r)**2) + reg_term
        return loss
    def mean_normalize(self,y,r):
        items=y.shape[0]
        users=y.shape[1]
        self.avg_items=np.zeros((items,1))  # the avg of a item liked by all users
        self.avg_users=np.zeros((users,1))  # the avg of a user who likes all items
        
        for i in range(items):
            idx = np.where(r[i,:] == 1)[0]
            if idx.shape[0]==0:
                self.avg_items[i]=0
            else:
                self.avg_items[i] = y[i,idx].mean()
        for i in range(users):
            idx = np.where(r[:,i] == 1)[0]
            if idx.shape[0]==0:
                self.avg_users[i]=0
            else:
                self.avg_users[i] = y[idx,i].mean()
        self.avg_users=self.avg_users.reshape(1,-1)
        # self.avg_items=np.average(y,axis=1).reshape(-1,1)
        # self.avg_users=np.average(y,axis=0).reshape(-1,1)

        # print(self.avg_items.shape)
        # print(self.avg_users.shape)
        # y=y-self.avg_items
        # print(self.avg_items)
        # print(self.avg_users)
        # print(y)
        # input("sd")

    def gradient_descent(self,X,y,r):
        h_x=self.predict(X,backend=True)
        # print(y)
        # print(y*r)
        # input("'sd")
        grads=np.matmul(X.T,(h_x-y)*r)  # (2,5)x(5,4)=(2,4) which is the shape of theta
        if not self.use_bias:
            reg_term=self.reg*self.theta
            grads+=reg_term
            self.theta-=self.lr*grads
        else:
            # reg_term=self.reg*self.theta[:-1,:]   # erroorr
            reg_term=self.reg*self.theta[1:,:]
            grads[1:,:]+=reg_term
            self.theta-=self.lr*grads

    def fit(self,item_features,ratings,r,iter=100):
        X=item_features.copy()
        y=ratings.copy()
        self.mean_normalize(y,r)
        if self.mean_normalization:
            y=y-self.avg_items
            # y=y-self.avg_users
        if self.use_bias:
            self.theta=np.random.randn(X.shape[-1]+1,y.shape[-1])*0.01 # +1 for bias term
            X=np.c_[np.ones((X.shape[0],1)),X]
        else:            
            self.theta=np.random.randn(X.shape[-1],y.shape[-1])*0.01
        # print(X.shape)
        # print(self.theta.shape)
        # pred=self.predict(X,backend=True)
        # print(pred.shape)
        # print(y.shape)
        for i in range(iter):
            self.gradient_descent(X,y,r)
            print("iteration:{:d} loss:{:.2f}".format(i+1,self.loss(X,y,r,backend=True)))



# r_s=content_based(use_bias=False)
# r_s=content_based()
r_s=content_based(mean_normalization=True)
# r_s=content_based()
# r_s.lr=0.0001
r_s.reg=0
r_s.fit(item_features,ratings,r,iter=10000)
print('loss:',r_s.loss(item_features,ratings,r))
print("original")
print(ratings)
preds=r_s.predict(item_features)
preds=preds*((ratings==-1)*1)
print("predictions")
print(np.around((ratings*r)+preds,2))
print("user avg rating:")
print(r_s.avg_items)
        