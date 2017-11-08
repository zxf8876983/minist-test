#-*- coding:UTF-8 -*-
'''
Created on 2017年5月22日

@author: zxf
'''
# from PIL import Image 
# 
# def Image_process(argv):
#     im = Image.open(argv)
#     #out=im.resize((28,28),Image.ANTIALIAS)
#     #out.show()
#     #out.save("sample.png")
#     tv = list(im.getdata())
#     tva = [ (255-x)*1.0/255.0 for x in tv]
#     print tva
#     
# if __name__ == "__main__":
#     Image_process("test_num/0_0.png")

import tensorflow as tf
import numpy as np

a = tf.placeholder("float")
b = tf.placeholder("float")
y = a * b
y2 = tf.multiply(a,b)
sess = tf.Session()
r, r2 = sess.run([y,y2], feed_dict={a:[3,1], b:[4,1]})
print r
print r2
sess.close()