'''
Created on 2017. 2. 15.

@author: archi45
'''

import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

y = tf.multiply(a, b)

with tf.Session() as sess:

    print(sess.run(y, feed_dict={a: 3., b: 2.}))