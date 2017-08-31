# -*- coding: utf-8 -*-
"""
MNIST use CNN
@author: yangjing
"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image
import os

mnist = input_data.read_data_sets("resources/MNIST_data/", one_hot=True)  # 下载并加载mnist数据
x = tf.placeholder(tf.float32, [None, 784])  # 输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 10])  # 输入的标签占位符


# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层
def max_pool(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建网络
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 转换输入数据shape,以便于用于网络中

# 第一个卷积层conv1 layer
weight_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
biases_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + biases_conv1)   # output size 28x28x32
tf.summary.histogram('conv1' +'/weights', weight_conv1)

# 第一个池化层 # output size 14x14x32
h_pool1 = max_pool(h_conv1)

# 第二个卷积层conv2 layer
weight_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
biases_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_conv2) + biases_conv2)  # output size 14x14x64
tf.summary.histogram('conv2' +'/weights', weight_conv2)
print weight_conv2

# 第二个池化层  # output size 7x7x64
h_pool2 = max_pool(h_conv2)

# func1 layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层
tf.summary.histogram('fc1'+'/weights', W_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout层

# func2 layer
weight_fc2 = weight_variable([1024, 10])
bias_fc2 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, weight_fc2) + bias_fc2)  # softmax层
tf.summary.histogram('fc2'+'/weights', weight_fc2)
tf.summary.histogram('y', y_predict);

# the error between prediction and real data
cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict))  # 交叉熵
tf.summary.scalar('loss_function', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # Adam算法的优化器
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

####TensorBoard
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/tmp/mnist_logs', session.graph)

# train
def train():
    for i in range(1000):
        batch = mnist.train.next_batch(50)

        summary_str = session.run(merged_summary_op, feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})
        summary_writer.add_summary(summary_str, i)

        if i % 100 == 0:  # 训练100次，验证一次
            train_acc = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 1.0})
            print('step', i, 'training accuracy', train_acc)
            train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

    test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
    print("test accuracy", test_acc)


# save variables
def save():
    saver = tf.train.Saver()
    saver.save(session, save_path)


# restore variables
def restore():
    saver = tf.train.Saver()
    saver.restore(session, save_path)


# close session
def close():
    session.close();


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
    nm = im_arr.reshape((1, 784))

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
            print(session.run(ans, feed_dict={x: oneTestx, keep_prob: 1}))


save_path = "resources/MNIST_data/model"

train()
save()
# restore()
# test_my_picture()
close()
