# coding:utf-8
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 获取mnist手绘图集
# MNIST_data/train-images-idx3-ubyte.gz 训练集图片：55000张训练图片，5000张验证图片
# MNIST_data/train-labels-idx1-ubyte.gz 训练集图片对应的数字标签
# MNIST_data/t10k-images-idx3-ubyte.gz  测试集图片：10000张图片
# MNIST_data/t10k-labels-idx1-ubyte.gz  测试集图片对应的数字标签
mnist = input_data.read_data_sets("resources/MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

# 初始化权值W
W = tf.Variable(tf.zeros([784, 10]))
# 初始化偏置项b
b = tf.Variable(tf.zeros([10]))

# 加权变换并进行softmax回归，得到预测概率
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)

# 求交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict), reduction_indices=1))

# 用梯度下降法使得残差最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 在测试阶段，测试准确度计算
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
# 多个批次的准确度均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 训练阶段，迭代1000次
    for i in range(55000):
        # 按批次训练，每批100行数据
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 执行训练
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})
        # 每训练100次，测试一次
        if (i % 100 == 0):
            print 'step', i, "accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})

    # print CNN.train.images.shape
    # print CNN.train.labels.shape
    # print CNN.validation.images.shape
    # print CNN.validation.labels.shape
    # print CNN.test.images.shape
    # print CNN.test.labels.shape
