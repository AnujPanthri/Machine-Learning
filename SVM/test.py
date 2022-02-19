from tabnanny import verbose
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create 3D Plot

# Colors definition
colors = {
0: '#b40426',
1: '#3b4cc0',
2: '#f2da0a',
3: '#fe5200'
# ... and so on
}

# Generate data
X, y = make_gaussian_quantiles(n_features=2, n_classes=2, n_samples=200, mean=(2,3))

# plt.plot(X[:,0],X[:,1],'bo')
# plt.show()

# Scale data
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
from data_preprocessing import *
f_s=feature_scaling(X)
X=f_s.transform(X)

# Generate Z component
# z = RBF(1.0).__call__(X)[0]
print(RBF(0.2).__call__(X)[0].shape)
z = RBF(0.5).__call__(X)[0]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = list(map(lambda x: colors[x], y))
ax.scatter(X[:, 0], X[:, 1], z, c=colors, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# import sklearn

from sklearn import svm

#Create a svm Classifier
# model = svm.SVC(kernel='linear')
model = svm.SVC(kernel='rbf',verbose=True)

model.fit(X,y)

y=y.reshape(-1,1)





x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
data_as_input=np.c_[xx.ravel(), yy.ravel()]
zz=model.predict(data_as_input)
# zz=(zz>1)*1
# zz=(zz>0)*1

zz=zz.reshape(xx.shape)

plt.figure(figsize=(10,7))
plt.title(f"Prediction(decision boundary)")
plt.plot(X[(y==0)[:,0],0],X[(y==0)[:,0],1],"go")
plt.plot(X[(y==1)[:,0],0],X[(y==1)[:,0],1],"bo")
plt.contourf(xx, yy, zz, cmap='Paired')
plt.xlabel('X')
plt.ylabel('y')
plt.show()





