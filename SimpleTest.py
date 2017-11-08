import numpy as np
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# batch_xs, batch_ys = mnist.test.next_batch(1)
# print batch_ys.shape
#np.savetxt('my_net/model_1/photo_data/x.txt',batch_xs)
#np.savetxt('my_net/model_1/photo_data/y.txt',batch_ys)
a=np.array([[1,2],[3,4]],dtype=np.int32)
print a.shape
np.savetxt("my_net/model_1/1.txt",a)
print np.loadtxt("my_net/model_1/1.txt").shape