import numpy as np
import copy
import matplotlib.pyplot as plt


def split_data_for_anomaly_detection(X,y,normal_split,anomaly_split):
    '''
    X:(m,n)
    y:(m,1)
    normal_split:(train,val,test) ex:(60,20,20) (ratio)
    anomaly_split:(val,test)      ex:(50,50)    (ratio)
    '''

    X=copy.deepcopy(X)
    y=copy.deepcopy(y)

    rand_num=np.random.random_integers(1,1000)
    np.random.seed(rand_num)
    np.random.shuffle(X)
    np.random.seed(rand_num)
    np.random.shuffle(y)

    normal_x,normal_y=X[(y==0)[:,0],:],y[(y==0)[:,0],:]
    anomaly_x,anomaly_y=X[(y==1)[:,0],:],y[(y==1)[:,0],:]
    
    #indexes for normal examples
    train_idx=int((normal_split[0]/100)*normal_x.shape[0])
    val_idx=train_idx+int((normal_split[1]/100)*normal_x.shape[0])
    test_idx=val_idx+int((normal_split[2]/100)*normal_x.shape[0])
    
    #adding normal examples
    train_x,train_y=normal_x[:train_idx,:],normal_y[:train_idx,:]
    val_x,val_y=normal_x[train_idx:val_idx,:],normal_y[train_idx:val_idx,:]
    test_x,test_y=normal_x[val_idx:test_idx,:],normal_y[val_idx:test_idx,:]

    # print("normal_examples")
    # print(train_x.shape,train_y.shape)
    # print(val_x.shape,val_y.shape)
    # print(test_x.shape,test_y.shape)

    #indexes for anomalous examples
    val_idx=int((anomaly_split[0]/100)*anomaly_x.shape[0])
    test_idx=val_idx+int((anomaly_split[1]/100)*anomaly_x.shape[0])

    #adding anomalous examples
    val_x,val_y=np.r_[ val_x,anomaly_x[:val_idx,:] ],np.r_[ val_y,anomaly_y[:val_idx,:] ]
    test_x,test_y=np.r_[ test_x,anomaly_x[val_idx:test_idx,:] ],np.r_[ test_y,anomaly_y[val_idx:test_idx,:] ]
    
    # 60/100*m
    # 20/100*m 
    # 20/100*m
    
    # print("all_examples")
    # print(train_x.shape,train_y.shape)
    # print(val_x.shape,val_y.shape)
    # print(test_x.shape,test_y.shape)

    return train_x,train_y,val_x,val_y,test_x,test_y

def show_data(X,y):
    plt.figure()
    plt.scatter(X[(y==0)[:,0],0],X[(y==0)[:,0],1],label='0')
    plt.scatter(X[(y==1)[:,0],0],X[(y==1)[:,0],1],label='1')
    plt.legend(loc='best')
    plt.show()



def show(model,X,y,threshold=None):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    data_as_input=np.c_[xx.ravel(), yy.ravel()]
    zz=model.predict(data_as_input)
    # zz=(zz>1)*1
    if threshold is not None:
        zz=(zz<threshold)*1
    
    zz=zz.reshape(xx.shape)

    plt.figure(figsize=(10,7))
    plt.title(f"Prediction(decision boundary)")
    plt.plot(X[(y==0)[:,0],0],X[(y==0)[:,0],1],"rx")
    plt.plot(X[(y==1)[:,0],0],X[(y==1)[:,0],1],"bo")
    plt.contourf(xx, yy, zz, cmap='Paired')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()


def visualize_feature(X):
    bins=np.linspace(0,np.max(X),20)
    plt.hist(X,bins=bins)

def feature_visualization(X):
    col=4
    n=X.shape[1]
    fig=plt.figure()
    for i in range(n):
        fig.add_subplot(int(np.ceil(n/col)),col,(i+1))
        visualize_feature(X[:,i])
    plt.show()