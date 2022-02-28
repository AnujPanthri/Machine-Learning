import numpy as np
import matplotlib.pyplot as plt
from data import *
from data_preprocessing import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# X,y,labels = load_rgb() # data loading
# X,y= datasets.make_blobs(n_samples=100,n_features=4,centers=4,random_state=142)
# X,y= datasets.make_blobs(n_samples=100,n_features=4,centers=4,random_state=12323)
# X,y= datasets.load_digits(return_X_y=True)
# X,y= datasets.load_iris(return_X_y=True)
X,y= datasets.load_wine(return_X_y=True)
y=y.reshape(-1,1)

print("X:",X.shape)
# X=X/255

from data_preprocessing import feature_scaling
scaler=feature_scaling(X)
X=scaler.transform(X)


# scaler=StandardScaler()
# scaler.fit(X)
# X=scaler.transform(X)

pca=PCA(2)
pca.fit(X)
print(sum(pca.explained_variance_ratio_))
X=pca.transform(X)

plt.figure()
plt.title("sklearn pca")
for i in np.unique(y):

    plt.scatter(X[(y==i)[:,0],0],X[(y==i)[:,0],1])
plt.savefig('sklearn.png',dpi=300)
plt.show()