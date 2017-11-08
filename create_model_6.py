#coding:utf-8
import os
import shutil
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
log_dir = '/home/zxf/文档/workspace/test/log/dir6'
def delete_files(path):
    if not os.listdir(path):
        pass
    else:
        shutil.rmtree(path)
        os.makedirs(path)
sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784],name='x')
y_ = tf.placeholder(tf.float32, [None, 10],name='labels')
#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
#y = tf.nn.softmax(tf.matmul(x, W) + b)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,name='W')
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,name='B')
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,28,28,1])
with tf.name_scope('conv1'):    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    tf.summary.histogram('weight', W_conv1)
    tf.summary.histogram('biases',b_conv1)    
    h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
    tf.summary.histogram('activation',h_conv1)
with tf.name_scope('pool1'):
    #h_conv1 = conv2d(x_image, W_conv1) + b_conv1
    h_pool1 = max_pool_2x2(h_conv1)
    
with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    tf.summary.histogram('weight', W_conv2)
    tf.summary.histogram('biases',b_conv2)  
    h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
    tf.summary.histogram('activation',h_conv2)
    #h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

    
with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
with tf.name_scope('softmax'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
# Define loss and optimizer
with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
"""
Train the model and save the model to disk as a model2.ckpt file
file is stored in the same directory as this python script is started

Based on the documentatoin at
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""

#saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
merged_summary=tf.summary.merge_all()
delete_files(log_dir+'/train')
train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
train_writer.add_graph(sess.graph)
#with tf.Session() as sess:
    #sess.run(init_op)
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        s,acc=sess.run([merged_summary,accuracy],feed_dict={x: batch[0], y_: batch[1]})
        print("step %d, training accuracy %g"%(i, acc))
        train_writer.add_summary(s,i)
    #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#save_path = saver.save(sess, "my_net/model_5/model5.ckpt")
#print ("Model saved in file: ", save_path)

print("test accuracy %g"%accuracy.eval(feed_dict={
    #x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    x: mnist.test.images, y_: mnist.test.labels}))



