#-*- coding:UTF-8 -*-
'''
Created on 2017年5月22日

@author: zxf
'''
import sys
import numpy as np
import tensorflow as tf
from PIL import Image,ImageFilter

def predictint(imvalue):
    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """
    
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    a_0 = tf.placeholder("float")
    a_1 = tf.placeholder("float")
    a_2 = tf.placeholder("float")
    #y = tf.matmul(x, W) + b
    y=tf.matmul(x, W) + b
    y_2=tf.mul(y,y)
    z = tf.nn.relu(y)
    new_z=0.1992 + 0.5002*y + 0.1997*y_2

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    """
    Load the model.ckpt file
    file is stored in the same directory as this python script is started
    Use the model to predict the integer. Integer is returend as list.

    Based on the documentatoin at
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
    """
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "my_net/model_1/model.ckpt")
        #print ("Model restored.")
   
#         prediction=tf.argmax(y,1)
#         return prediction.eval(feed_dict={x: [imvalue]}, session=sess)
        print z.eval(feed_dict={x: [imvalue]}, session=sess)
        print new_z.eval(feed_dict={x: [imvalue]}, session=sess)
def Image_process(argv):
    im = Image.open(argv)
    #out=im.resize((28,28),Image.ANTIALIAS)
    #out.show()
    im.save("sample.png")
    tv = list(im.getdata())
    tva = [ (255-x)*1.0/255.0 for x in tv]
    return tva

argv="test_num/0_0.png"
imvalue=Image_process(argv)
predictint(imvalue)
  




