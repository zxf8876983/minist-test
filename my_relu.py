import numpy as np

def relu(x):
    y=np.zeros(x.size,np.float)
    for i in range(x.size):
        if x[i]>0:
            y[i]=x[i]
    return y
