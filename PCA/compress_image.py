import numpy as np
import PIL
import matplotlib.pyplot as plt
import glob
from pca import PCA



size=20
img_list=glob.glob('./*.jpg')
X=np.zeros((0,size,size,3))
for img_path in img_list:
    img=np.array(PIL.Image.open(img_path).convert("RGB").resize((size,size))).reshape(1,size,size,3)
    X=np.r_[X,img]
    
X=X.astype('uint8')
print(X.shape)

def show(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

all_X=X.copy()/255
X=X[0,...]
# show(X[0,...])
# X=X/255

r,g,b=X[:,:,0],X[:,:,1],X[:,:,2]

var=0.98

r_pca=PCA()
var=r_pca.fit(r,var=var)
# print("variance:",var)
print("K:",r_pca.K)
X_reduced=r_pca.transform(r)
# print(X_reduced.shape)
r_approx=r_pca.transform_back(X_reduced)
# r_approx=np.around(r_approx,2)
# r_approx=r_approx.astype('uint8')
# print("R:",(r-r_approx).sum())
# print("R:",r[-5:,-5:])
# print("approx",r_approx[-5:,-5:])
# print("R:",(r[-5:,-5:]-r_approx[-5:,-5:]))

g_pca=PCA()
var=g_pca.fit(g,var=var)
# print("variance:",var)
print("K:",g_pca.K)
X_reduced=g_pca.transform(g)
# print(X_reduced.shape)
g_approx=g_pca.transform_back(X_reduced)

# g_approx=np.around(g_approx,2)
# g_approx=g_approx.astype('uint8')
# print("G:",(g-g_approx).sum())

b_pca=PCA()
var=b_pca.fit(b,var=var)
# print("variance:",var)
print("K:",b_pca.K)
X_reduced=b_pca.transform(b)
# print(X_reduced.shape)
b_approx=b_pca.transform_back(X_reduced)
# b_approx=np.around(b_approx,2)
# b_approx=b_approx.astype('uint8')
# print("B:",(b-b_approx).sum())

m=X.shape[0]
r_approx=np.expand_dims(r_approx,axis=-1)
g_approx=np.expand_dims(g_approx,axis=-1)
b_approx=np.expand_dims(b_approx,axis=-1)

X_approx=np.concatenate([r_approx,g_approx,b_approx],axis=-1)
print("X_approx:",X_approx.shape)





# X=X*255
# X_approx=X_approx*255

X_approx=np.around(X_approx,2)
X_approx=X_approx.astype('uint8')
X=X.astype('uint8')
print(np.all(X_approx==X))
# print(X[:4,:4,:])
# print("see:",X_approx[:4,:4,:])


fig=plt.figure()
fig.add_subplot(1,2,1)
plt.title('img')
plt.axis('off')
plt.imshow(X)

fig.add_subplot(1,2,2)
plt.title('approx')
plt.axis('off')
plt.imshow(X_approx)
plt.show()
# fig=plt.figure()
# for i in range(m):
#     fig.add_subplot(m,2,(i*2)+1)
#     plt.title('img')
#     plt.axis('off')
#     plt.imshow(X[i,...])

#     fig.add_subplot(m,2,(i*2)+2)
#     plt.title('approx')
#     plt.axis('off')
#     plt.imshow(X_approx[i,...])
# plt.show()