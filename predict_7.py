# Copyright 2016 Niek Temme. 
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

"""Predict a handwritten integer (MNIST expert).

Script requires
1) saved model (model2.ckpt file) in the same location as the script is run from.
(requried a model created in the MNIST expert tutorial)
2) one argument (png file location of a handwritten integer)

Documentation at:
http://niektemme.com/ @@to do
"""

#import modules
import sys
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from PIL import Image, ImageFilter

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
def predictint():
    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """
    
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
       
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   
    
    W_conv1 = weight_variable([5, 5, 1, 1])
    b_conv1 = bias_variable([1])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 1, 1])
    b_conv2 = bias_variable([1])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 , 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #keep_prob = tf.placeholder(tf.float32)

    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
   
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "my_net/model_7/model7.ckpt")
        batch_xs, batch_ys = mnist.test.next_batch(10)
        prediction = tf.argmax(y_conv, 1)
        y_ = tf.argmax(batch_ys, 1)
        result, real_y, y_value, b_value = sess.run([prediction, y_, y_conv, b], feed_dict={x: batch_xs})
        #print ("Model restored.")
       
        #prediction=tf.argmax(y_conv,1)
        #return prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)
        W_conv1_value=W_conv1.eval()
        W_conv1_value=np.reshape(W_conv1_value,[5,5])
        #np.savetxt('my_net/model_7/W.txt',W_conv1_value)
        print W_conv1_value
        #W_conv1_value=tf.reshape(W_conv1_value,[5,5])
        #print W_conv1_value[0,1]
predictint()


