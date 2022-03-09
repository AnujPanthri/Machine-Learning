import sklearn.datasets as ds
import numpy as np
from helper import *
from confusion_matrix import *
from anomaly_detection import *
from data import *
from data_preprocessing import *


# X,y=load_heart_disease()
X,y,labels=load_credit_card()
print(labels)

# visualize_feature(X[:,1])
# feature_visualization(X)
# scaler=feature_scaling(X)
# X=scaler.transform(X)

print(X.shape,y.shape)
# train_x,train_y,val_x,val_y,test_x,test_y=split_data_for_anomaly_detection(X,y,normal_split=(60,20,20),anomaly_split=(50,50))
train_x,train_y,val_x,val_y,test_x,test_y=split_data_for_anomaly_detection(X,y,normal_split=(80,10,10),anomaly_split=(50,50))
# X=np.random.rand(1000,23)
# show_data(X,y)
# show_data(train_x,train_y)
# show_data(val_x,val_y)
# show_data(test_x,test_y)

model=anomaly_detection()
# from anomaly_detection_multivariate import *
# model=anomaly_detection_multivariate()
model.fit(train_x)
# show(model,test_x,test_y)

f1_score=model.find_e(val_x,val_y)
# f1_score=model.find_e_v2(val_x,val_y)
print("f1_score:",f1_score)
print("e:",model.e)


# y_pred=model.predict(train_x)
y_pred=model.predict(val_x)
all=f1(confusion(val_y,y_pred,2))
print("VAL:",all)

y_pred=model.predict(test_x)
all=f1(confusion(test_y,y_pred,2))
print("TEST:",all)

# show(model,test_x,test_y)

# 

# show(model,train_x,train_y)
# show(model,train_x,train_y,threshold=0.1)
# show(model,test_x,test_y,threshold=0.1)

