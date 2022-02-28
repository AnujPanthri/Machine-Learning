import numpy as np
import matplotlib.pyplot as plt
from data import *
from data_preprocessing import *
from pca import PCA
from sklearn import datasets



# X,y,labels = load_rgb() # data loading
# X,y = load_heart_disease() # data loading
# X,y= datasets.make_blobs(n_samples=100,n_features=4,centers=4,random_state=142)
# X,y= datasets.make_blobs(n_samples=100,n_features=4,centers=4,random_state=12323)
# X,y= datasets.load_digits(return_X_y=True)
# X,y= datasets.load_iris(return_X_y=True)
X,y= datasets.load_wine(return_X_y=True)
y=y.reshape(-1,1)

# X,y,_,_=split_data(X,y,split=1)

print("X:",X.shape)


# X=X/255

from data_preprocessing import feature_scaling
scaler=feature_scaling(X)
X=scaler.transform(X)



# fig=plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(X[:,0], X[:,1], X[:,2], c=y)
# plt.show()

pca=PCA(K=2)
var=pca.fit(X)
print("K :",pca.K)
print("variance :",var)
X=pca.transform(X)


# fig=plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(X[:,0], X[:,1], X[:,2], c=y)
# plt.show()

plt.figure()
plt.title("my pca")
for i in np.unique(y):

    plt.scatter(X[(y==i)[:,0],0],X[(y==i)[:,0],1])
plt.savefig('my.png',dpi=300)
plt.show()