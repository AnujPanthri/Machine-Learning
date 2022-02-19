import numpy as np

class kernel:
    def fit():
        raise NotImplementedError
    def forward():
        raise NotImplementedError
    def num_of_features():
        raise NotImplementedError

class  linear_kernel(kernel):
    def fit(self,x):
        self.support=x
    def num_of_features(self):
        # return self.support.shape[-1]
        return self.support.shape[0]
    def forward(self,x,support=False):
        if isinstance(support,np.ndarray):
            self.support=support
        x=np.matmul(x,self.support.T)
        # print(x.shape)
        # x=np.c_[np.ones_like(x[...,0:1]),x]
        return x
        

class  rbf_kernel(kernel):   # rbf(radial basic function)
    def __init__(self,sigma=3):
        self.sigma=sigma
    def fit(self,x):
        self.support=x
    def num_of_features(self):
        return self.support.shape[0]
    def forward(self,x,support=False):
        '''
        x=(any number,2)           (any_num,1,2)
        support=(m,2)               (1,m,2)
        out=(any number,m)
        '''
        if isinstance(support,np.ndarray):
            self.support=support
        out=np.exp(-1*np.sum(np.abs(x[:,None,:]-self.support[None,:,:])**2,axis=-1) / (2*(self.sigma**2)) )
        # out=np.c_[np.ones_like(out[...,0:1]),out]
        return out