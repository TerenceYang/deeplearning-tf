#!/usr/bin/env python
# -*- coding=utf-8 -*-
# @author: terence
# @date: 2017-07-04
# @description: implement a CNN model upon MNIST handwritten digits
# @ref: http://yann.lecun.com/exdb/mnist/

import gzip
import struct
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from PIL import Image
import os
import time


# MNIST data is stored in binary format,
# and we transform them into numpy ndarray objects by the following two utility functions
def read_image(file_name):
    with gzip.open(file_name, 'rb') as f:
        buf = f.read()
        index = 0
        magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
        index += struct.calcsize('>IIII')

        image_size = '>' + str(images * rows * columns) + 'B'
        ims = struct.unpack_from(image_size, buf, index)

        im_array = np.array(ims).reshape(images, rows, columns)
        return im_array


def read_label(file_name):
    with gzip.open(file_name, 'rb') as f:
        buf = f.read()
        index = 0
        magic, labels = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')

        label_size = '>' + str(labels) + 'B'
        labels = struct.unpack_from(label_size, buf, index)

        label_array = np.array(labels)
        return label_array


print "Start processing MNIST handwritten digits data..."
train_x_data = read_image("resources/MNIST_data/train-images-idx3-ubyte.gz")  # shape: 60000x28x28
train_x_data = train_x_data.reshape(train_x_data.shape[0], train_x_data.shape[1], train_x_data.shape[2], 1).astype(
    np.float32)
train_y_data = read_label("resources/MNIST_data/train-labels-idx1-ubyte.gz")
test_x_data = read_image("resources/MNIST_data/t10k-images-idx3-ubyte.gz")  # shape: 10000x28x28
test_x_data = test_x_data.reshape(test_x_data.shape[0], test_x_data.shape[1], test_x_data.shape[2], 1).astype(
    np.float32)
test_y_data = read_label("resources/MNIST_data/t10k-labels-idx1-ubyte.gz")

train_x_minmax = train_x_data / 255.0
test_x_minmax = test_x_data / 255.0

# Of course you can also use the utility function to read in MNIST provided by tensorflow
# from tensorflow.examples.tutorials.CNN import input_data
# CNN = input_data.read_data_sets("MNIST_data/", one_hot=False)
# train_x_minmax = CNN.train.images
# train_y_data = CNN.train.labels
# test_x_minmax = CNN.test.images
# test_y_data = CNN.test.labels

# Reformat y into one-hot encoding style
lb = preprocessing.LabelBinarizer()
lb.fit(train_y_data)
train_y_data_trans = lb.transform(train_y_data)
test_y_data_trans = lb.transform(test_y_data)

print "Start evaluating CNN model by tensorflow..."

# Model input
x = tf.placeholder(tf.float32, [None, 784])  # 输入的数据占位符
y_actual = tf.placeholder(tf.float32, [None, 10])  # 输入的标签占位符

x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])  # 转换输入数据shape,以便于用于网络中


# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and Pooling
def conv2d(x, W):
    # `tf.nn.conv2d()` computes a 2-D convolution given 4-D `input` and `filter` tensors
    # input tensor shape `[batch, in_height, in_width, in_channels]`, batch is number of observation
    # filter tensor shape `[filter_height, filter_width, in_channels, out_channels]`
    # strides: the stride of the sliding window for each dimension of input.
    # padding: 'SAME' or 'VALID', determine the type of padding algorithm to use
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # `tf.nn.max_pool` performs the max pooling on the input
    #  ksize: the size of the window for each dimension of the input tensor.
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# First convolutional layer
# Convolution: compute 32 features for each 5x5 patch
# Max pooling: reduce image size to 14x14.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
# Convolution: compute 64 features for each 5x5 patch
# Max pooling: reduce image size to 7x7
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
# Fully-conected layer with 1024 neurons
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# To reduce overfitting, we apply dropout before the readout layer.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and evaluate
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
y_predict = tf.nn.softmax(y_conv)
loss = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(1e-4)
# optimizer = tf.train.GradientDescentOptimizer(1e-4)
train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# train
def train():
    last_time = time.time();
    for step in range(2000):
        sample_index = np.random.choice(train_x_minmax.shape[0], 50)
        batch_xs = train_x_minmax[sample_index, :]
        batch_ys = train_y_data_trans[sample_index, :]
        if step % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x_image: batch_xs, y_actual: batch_ys, keep_prob: 1.0})
            cost_time = time.time() - last_time;
            print "cost time:%s, step %d, training accuracy %g" % (cost_time, step, train_accuracy)
        sess.run(train_step, feed_dict={x_image: batch_xs, y_actual: batch_ys, keep_prob: 0.5})

    print "test accuracy %g" % sess.run(accuracy, feed_dict={
        x_image: test_x_minmax, y_actual: test_y_data_trans, keep_prob: 1.0})


# save variables
def save():
    saver = tf.train.Saver()
    saver.save(sess, save_path)


# restore variables
def restore():
    saver = tf.train.Saver()
    saver.restore(sess, save_path)


# close session
def close():
    sess.close();


# get the test picture
def get_test_pic_array(filename):
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)

    im_arr = np.array(out.convert('L'))

    num0 = 0
    num255 = 0
    threshold = 100

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1

    if (num255 > num0):
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if (im_arr[x][y] < threshold):  im_arr[x][y] = 0
                # if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
                # else : im_arr[x][y] = 255
                # if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

    out = Image.fromarray(np.uint8(im_arr))
    split_names = filename.split('/')
    out.save(split_names[0] + '/28pix/' + split_names[split_names.__len__() - 1])
    # print im_arr
    nm = im_arr.reshape((1, 28, 28, 1))

    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)

    return nm


# Test my picture
# 代码来自Tensorflow中手写数字识别中基于卷积神经网络模型的例子，在例子的基础上修改部分，可以测试自己的图片。
# 将待测图片放在testPicture文件夹，在testPicture/28pix中将保存待测图片的28*28灰度图片
# 例如：testPicture/test/mnist_test_9.png
# testPicture/test/test_0_1.png
def test_my_picture():
    # testNum = input("input the number of test picture:")
    files = os.listdir("resources/testpicture/test")
    test_num = files.__len__()
    for i in range(test_num):
        # testPicture = raw_input("input the test picture's path:")
        if (files[i].startswith("test")):
            print("The number is:" + files[i])
            test_picture = "resources/testpicture/test/" + files[i]
            oneTestx = get_test_pic_array(test_picture)
            ans = tf.argmax(y_predict, 1)
            print("The prediction answer is:")
            print(sess.run(ans, feed_dict={x_image: oneTestx, keep_prob: 1}))


save_path = "resources/MNIST_data/model"

train()
save()
# restore()
# test_my_picture()
close()