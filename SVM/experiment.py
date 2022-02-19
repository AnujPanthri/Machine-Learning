import numpy as np
from kernels import *
import matplotlib.pyplot as plt


X=np.array([[1,1],
            [3,1],
            [2,3],
            [3,3],
            [2,5],
            [4,5],
            # [2,5],
            ])

plt.scatter(X[:,0],X[:,1])
plt.show()

# kernel=rbf_kernel(1)
kernel=rbf_kernel()
kernel.fit(X)
out=kernel.forward(X)

print(out)
i,j=2,3
print("should be same:",out[i,j],out[j,i])