import numpy as np
# from data_preprocessing import feature_scaling


class PCA:
    def __init__(self,K=None):
        if K is not None:
            self.K=K
    def fit(self,X,var=None):
        ''' X is the data(which should be normalized)
            var: give this value to reserve that much variance ex var=0.98
        '''
        # self.scaler=feature_scaling(X)
        # X=self.scaler.transform(X)
        # print("X",X.shape)

        self.datashape=X.shape
        # first convert X to (m,n) shape
        X=X.reshape(self.datashape[0],-1)
        self.datashape=self.datashape[1:]   # removing m from the shape
        # print(X.shape)
        # to apply pca first calculate Covariance matrix which is (n,n)
        covariance_matrix=np.matmul(X.T,X)  # (n,n)
        u,s,_=np.linalg.svd(covariance_matrix) #(n,n)
        # print("s:",s.shape)
        if var is not None:
            for k in range(1,s.shape[0]+1):
                variance=np.sum(s[:k])/np.sum(s)        
                # print(variance)
                if variance>=var:
                    self.K=k
                    break
        self.u=u[:,:self.K]   # (n,k)
        variance=np.around(np.sum(s[:self.K])/np.sum(s),4)
        return variance
    def transform(self,X):
        # X=self.scaler.transform(X)
        X=X.reshape(X.shape[0],-1)   
        out= np.matmul(X,self.u)  # (m,n)*(n,k)=(m,k)
        return out
    def transform_back(self,X):
        X=X.reshape(X.shape[0],-1)   
        X_approx=np.matmul(X,self.u.T)  # (m,k)*(k,n)=(m,n)
        X_approx=X_approx.reshape((X.shape[0],)+self.datashape)
        # X_approx=self.scaler.transform_to_normal(X_approx)
        return X_approx

