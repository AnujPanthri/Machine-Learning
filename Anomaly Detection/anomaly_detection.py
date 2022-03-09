# probability distribution
import sklearn.datasets as ds
import numpy as np
from helper import *
from confusion_matrix import *

class anomaly_detection:
    def __init__(self):
        self.e=None

    def fit(self,X):

        m=X.shape[0]
        self.mean=(1/m)*np.sum(X,axis=0) # (n) values for mean of each feature
        self.mu=(1/m)* np.sum((X-self.mean.reshape(1,-1))**2,axis=0) # (n) variance of each feature

        self.mean=self.mean.reshape(1,-1)
        self.mu=self.mu.reshape(1,-1)
        # print(self.mean.shape)
        # print(self.mu.shape)
    def find_e_v2(self,X,y): 
        # search_space=np.linspace(0,0.1,40)
        # print(search_space)
        best_f1=0
        best_e=0
        self.e=0
        update=0.001
        # lr=0  # lower range
        ur=None
        test_e=0
        searching_best=False
        while True:
            
            self.e=test_e
            pred=self.predict(X)
            f1_scores=f1(confusion(y,pred,2))[-2] # scores for both classes 0 and 1
            f1_score=np.average(f1_scores)
            # print(f1_scores)
            if f1_scores[1]==0: # [0.6 , 0]  all zeros
                lr=test_e
            elif f1_scores[0]==0: # [0 , 0.8]  all ones
                ur=test_e
                update=(ur-lr)/1000
                test_e=lr
                searching_best=False
                # print("range:",lr,ur)
                # print("update:",update)
            else:
                if f1_score>best_f1:
                    best_f1=f1_score
                    searching_best=True
                    print("best_f1:",best_f1,"at e:",test_e)
                    best_e=test_e
                elif searching_best==True:
                    searching_best=False
                    ur=test_e
                    update=(ur-lr)/100 # increase 100 to do more fine search
                    test_e=lr
                    searching_best=False
                    # print("range:",lr,ur)
                    # print("update:",update)
                else:
                    lr=test_e

            if ur is not None:
                # print("diff:",ur-lr)
                if ur-lr<1e-40:
                    break
            test_e=test_e+update
            # if f1_score>best_f1:
            #     best_f1=f1_score
            #     best_e=test_e
            #     lr=test_e
            #     print(f1_scores)
            #     print(f1_score)
            # elif f1_score<=best_f1:
            #     ur=test_e
            #     update=(ur-lr)/10
            #     print("range:",lr,ur)
            #     print("update:",update)

            # print(f1_scores)
            # print(f1_score)
        self.e=best_e
        return best_f1

    def find_e(self,X,y):
        # search_space=np.linspace(0,1,40)
        # search_space=[0.0000e+00, 1.0527717316e-70, 1.0527717316e-50, 1.0527717316e-14,1.0527717316e-7,1.0527717316e-4,1.0527717316e-2,1.0527717316e-1,1.0527717316]
        pval=self.predict(X)
        stepsize=(np.max(pval) - np.min(pval))/1000
        # search_space=np.arange(np.min(pval),np.max(pval),stepsize)
        # print(search_space)
        best_f1=0
        best_e=0
        self.e=0
        test_e=0
        found_best=False
        # for test_e in search_space:
        while test_e<np.max(pval):
            self.e=test_e
            pred=self.predict(X)
            f1_scores=f1(confusion(y,pred,2))[-2] # scores for both classes 0 and 1
            f1_score=np.average(f1_scores)
            # print(f1_scores,end='\n')
            if f1_score>best_f1 and f1_scores[0]!=0 and f1_scores[1]!=0:
                best_f1=f1_score
                best_e=test_e
                found_best=True
                # print("e:",test_e,"f1:",f1_score,end='\n')
            
            # elif found_best==True:
            #     break
            test_e+=stepsize # update
        self.e=best_e
        return best_f1

    def predict(self,X):

        p_x=(1/(np.sqrt((2*np.pi))*self.mu))*np.exp(-((X-self.mean)**2)/(2*(self.mu**2) ) )
        p_x=np.prod(p_x,axis=-1)
        p_x=p_x.reshape(-1,1)
        # print("P_x:",p_x.shape)
        if self.e is not None:
            p_x=(p_x<self.e)*1
        return p_x


