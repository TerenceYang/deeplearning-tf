# coding:utf-8
import tensorflow as tf

x = tf.Variable(tf.ones([2, 3]))
y = tf.Variable(tf.ones([3, 2]))

z = tf.matmul(5*x, 4*y)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(z))
