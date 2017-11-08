#-*- coding: UTF-8 -*-
# Copyright 2016 Niek Temme.
# Adapted form the on the MNIST biginners tutorial by Google. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
Documentation at
http://niektemme.com/ @@to do

This script is based on the Tensoflow MNIST beginners tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
"""

#import modules
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#t = tf.nn.relu(tf.matmul(x, W) + b)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init_op = tf.initialize_all_variables()
saver = tf.train.Saver()


# Train the model and save the model to disk as a model.ckpt file
# file is stored in the same directory as this python script is started
"""
The use of 'with tf.Session() as sess:' is taken from the Tensor flow documentation
on on saving and restoring variables.
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                #x:batch[0], y_: batch[1], keep_prob: 1.0})
                x:batch_xs, y_: batch_ys})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    print("test accuracy %g"%accuracy.eval(feed_dict={
        #x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        x: mnist.test.images, y_: mnist.test.labels}))
#         if i==9999:
#             W_val,b_val=sess.run([W,b])
#             print W_val,b_val
# for var in tf.trainable_variables():
#     print var
    #b_val=b.eval()
    #W_val=W.eval()
    #print W_val
    #print b_val
    #print b_val.shape
    #save_path = saver.save(sess, "my_net/model_1/model.ckpt")
    #np.savetxt('my_net/model_1/photo_data/W.txt',W_val)
    #np.savetxt('my_net/model_1/photo_data/b.txt',b_val)
    #np.savetxt('my_net/model_1/B.txt',b_val)
    #print ("Model saved in file: ", save_path)

