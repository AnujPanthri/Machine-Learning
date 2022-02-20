import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import datasets



class k_means:
    def __init__(self,K=3):
        self.K=K
        self.mu=None
    def random_clusters(self,x):
        # idx=np.random.choice(np.arange(0,x.shape[0]),self.K) # don't use cuz it can give repeatition
        idx=random.sample(range(0,x.shape[0]),self.K)     # no repeatition
        self.mu=x[idx,:]
    def x_cluster_idx(self,x,return_nearest_idx=False):
        '''
        self.mu (K,2)   generally (K,n)
        self.X  (m,2)   generally (m,n)
        
        So we do this:

        self.mu (K,1,2)
        self.X  (1,m,2)

        C (K,m)
        C (,m)
        '''
        m,n=x.shape
        C=np.sqrt(np.sum((self.mu.reshape(self.K,1,n)-x.reshape(1,m,n))**2,axis=-1))
        class_id=np.argmin(C,axis=0)
        if return_nearest_idx:
            nearest_idx=np.argmin(C,axis=1)
            return class_id,nearest_idx
        return class_id

    def loss(self,x):
        if self.mu is None:
            self.random_clusters(x)
        # euclidian distance between cluster point k and its points
        
        C=self.x_cluster_idx(x)
        m,n=x.shape
        cost=0
        
        for k in range(self.K):
            k_x_distances=np.sqrt(np.sum((self.mu[k,:].reshape(1,n)-x[(C==k),:])**2,axis=-1)) # gives distances regarding k cluster(1,c belonging to k)
            cost+=np.sum(k_x_distances)
        cost=cost/m
        return cost
        

    def train(self,x,show=True,showat=100):
        loss_list=[]
        if self.mu is None:
            self.random_clusters(x)
        i=0
        while True:
            # find x(i) belongs to which cluster
            C=self.x_cluster_idx(x)
            while 0 in [np.sum((C==k)*1) for k in range(self.K)]:
                self.random_clusters(x)
                print('fix error:')
                C=self.x_cluster_idx(x)
            # while 0 in [np.sum((C==k)*1) for k in range(self.K)]:
            #     for k in range(self.K):
            #         if np.sum((C==k)*1)==0:
            #                 print('fixing error:',k+1)
            #                 self.mu[k,:]=x[np.random.choice(range(x.shape[0]),1),:]
            #                 C=self.x_cluster_idx(x)
            # removed=0
            # for k in range(self.K):
            #     if np.sum((C==k)*1)==0:
            #             print('fix error:',k+1)
            #             if k+1==self.K:
            #                 print(self.mu[k+1:,:].shape)
            #                 self.mu=self.mu[:k,:]
            #             else:
            #                 self.mu=np.r_[self.mu[:k,:],self.mu[k+1:,:]]
            #             removed+=1
            #             print(self.K,self.mu.shape)
            # self.K-=removed
            # C=self.x_cluster_idx(x)

            # shift the cluster to it's mean point
            old_mu=self.mu.copy()
            for k in range(self.K):
                if np.sum((C==k)*1)==0:
                        print('error:',k+1)
                x_from_k=x[(C==k),:]
                self.mu[k,:]=x_from_k.sum(axis=0)/x_from_k.shape[0]
            loss_list.append(self.loss(x))
            if show:
                if (i+1)%showat==0:
                    print("iteration:{:3d} loss:{:.4f}".format((i+1),self.loss(x)))
            if not (self.mu-old_mu).any():
                # print((self.mu-old_mu))
                if show:
                    print('K-means have converged')
                break
            i+=1
        return loss_list
    def random_initializations(self,x,num_of_models=100):
        mu_list=[]
        loss_list=[]
        for i in range(num_of_models):
            self.random_clusters(x)
            mu_list.append(self.mu)
            self.train(x,show=False)
            loss_list.append(self.loss(x))
        idx=np.argmin(loss_list)
        self.mu=mu_list[idx]

    def elbow_method(self,x,K=10):
        original_k=self.K
        k_list=list(range(1,K+1))
        loss_list=[]
        for k in k_list:
            print('with ',k,' clusters:')
            self.K=k
            self.random_clusters(x)
            self.random_initializations(x,num_of_models=10)
            self.train(x,show=False)
            loss_list.append(self.loss(x))
        plt.figure()
        plt.title('Elbow Method')
        plt.plot(k_list,loss_list)
        plt.xlabel('num of clusters K')
        plt.ylabel('Loss')
        plt.show()
        self.K=original_k
    def loss_graph(self,loss_list):
        plt.figure()
        plt.plot(range(1,len(loss_list)+1),loss_list)
        plt.title('loss graph')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.savefig('loss graph.png')
        plt.show()
    def show_2D_data(self,x,show=True,title=None):
        if self.mu is None:
            self.random_clusters(x)
        C=self.x_cluster_idx(x)

        plt.figure()
        for k in range(self.K):

            plt.scatter(x[(C==k),0],x[(C==k),1],label=f'data from cluster {k+1}')

            plt.scatter(self.mu[k,0],self.mu[k,1],label=f'cluster {k+1}')
        plt.legend(loc='best')
        
        if title is not None:
            plt.title(title)
            plt.savefig(f'{title}.png')
        if show:
            plt.show()


if __name__=='__main__':

    from scipy.io import loadmat
    mat = loadmat("ex6data1.mat")
    X = mat["X"]
    y = mat["y"]
    y=y.reshape(-1)
    # X,y=datasets.make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=10)
    # X,_=datasets.make_moons(n_samples=100)

    # plt.figure()
    # plt.scatter(X[(y==0),0],X[(y==0),1])
    # plt.scatter(X[(y==1),0],X[(y==1),1])
    # plt.show()

    print("X:",X.shape)

    model=k_means(K=2)

    model.show_2D_data(X,title='before_training')

    # model.random_initializations(X,iter=10,num_of_models=10)
    # model.elbow_method(X,K=10)
    # model.K=4
    # model.random_clusters(X)

    hist=model.train(X,showat=1)
    model.loss_graph(hist)

    model.show_2D_data(X,title='after_training')

