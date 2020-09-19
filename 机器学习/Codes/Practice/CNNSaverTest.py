## ---------CNN Saver Test---------##

import tensorflow as tf
import numpy as np
import random
import time
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    # 使用测试数据集进行预测结果，预测值是 1* 10 的， 每个元素是一个概率，值越大代表可能性越大，最大值的位置即预测的 数字
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})

    # 使用测试数据集的准确结果，与真实值对比（有正确也有错误）
    # tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))

    # 计算准确率，即计算 correct_prediction 中有多少正确以及多少错误
    # tf.cast强制类型转换，将 correct_prediction 布尔型的转为 tf.float32
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # sess.run()
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1, 中间两个1代表padding时在x方向运动一步，y方向运动一步
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

## POOLING 处理 strides 步长较大的问题
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # pooling 有两种，一种是最大值池化，一种是平均值池化，本例采用的是最大值池化tf.max_pool()。池化的核函数大小为2x2，
    # 因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 784], name="x_input") # 28x28
    ys = tf.placeholder(tf.float32, [None, 10], name="y_input")
    keep_prob = tf.Variable((1), dtype=tf.float32, name="keep_prob")

## 由于在 上述函数中输入的 x 参数，shape=[batch, height, width, channels],故需要处理一下 xs
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # print(x_image.shape)  # [n_samples, 28,28,1]

##---建立卷积层---##
## conv1 layer ##
with tf.name_scope("conv1_layer"):
    with tf.name_scope("conv1_Weights"):
        # patch 5x5, in size 1, out size 32, 原本数据是黑白，高度只有1，现在输出 32 是高度增加，和原理一样的操作
        W_conv1 = tf.Variable(np.arange(5*5*1*32).reshape((5, 5, 1, 32)), dtype=tf.float32, name="conv1_layer_weight")
    with tf.name_scope("conv1_bias"):
        b_conv1 = tf.Variable((32), dtype=tf.float32, name="conv1_layer_bias")
    # conv1 + 激励函数
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32, W_conv1输出的高度已经发生改变为 32
    # 池化 option
    h_pool1 = max_pool_2x2(h_conv1)            # output size 14x14x32， 池化是2*2，等于宽度和长度缩小1倍

## conv2 layer ##
with tf.name_scope("conv2_layer"):
    with tf.name_scope("conv2_Weights"):
        W_conv2 = tf.Variable(np.arange(5*5*32*64).reshape((5, 5, 32, 64)), dtype=tf.float32, name="conv2_layer_weight")
        # patch 5x5, in size 32, out size 64 ，从第一次的 conv1 高度 32--->> 64 继续变高
    with tf.name_scope("conv2_bias"):
        b_conv2 = tf.Variable((64), dtype=tf.float32, name="conv2_layer_bias")
    # conv1 + 激励函数
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
    # 池化 option
    h_pool2 = max_pool_2x2(h_conv2)          # output size 7x7x64， 池化是2*2，等于宽度和长度再缩小1倍

##---建立全连接层---##
## fc1 layer ##
# 基本神经网络搭建 相当于之前的 add_layer, 也是根据卷积网络的步骤进行的

# 此时weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64， 后面的输出size我们继续扩大，定为1024
with tf.name_scope("FC1_layer"):
    with tf.name_scope("FC1_Weights"):
        W_fc1 = tf.Variable(np.arange(7*7*64*1024).reshape((7*7*64, 1024)), dtype=tf.float32, name="FC1_layer_weight")
    with tf.name_scope("FC1_bias"):
        b_fc1 = tf.Variable((1024), dtype=tf.float32, name="FC1_layer_bias")

    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    ## overfitting option
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
# 最终输出是 0-9 的概率，只有 10 个
with tf.name_scope("FC2_layer"):
    with tf.name_scope("FC2_Weights"):
        W_fc2 = tf.Variable(np.arange(1024*10).reshape((1024, 10)), dtype=tf.float32, name="FC2_layer_weight")
    with tf.name_scope("FC2_bias"):
        b_fc2 = tf.Variable((10), dtype=tf.float32, name="FC2_layer_bias")
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope("loss1"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
    # 将 loss 曲线展现出来
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

from practice.util import logdirs
path="H:\\software\\pycharm\\Codes\\practice\\"
logdirs(path+"log111s")
test_writer = tf.summary.FileWriter("log111s/", sess.graph)

saver = tf.train.Saver()
saver.restore(sess, "CNNSaver/cnnsave_net.ckpt")
print("weights:", sess.run(W_conv1))
print("biases:", sess.run(b_conv1))

timecurrent = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
print(timecurrent)
# 开始训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(i, end=" ")
        timecurrent = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())) + '_' + str(random.randint(1000, 9999))
        print(timecurrent, end=" ")
        test=sess.run(merged, feed_dict={xs: mnist.test.images[:1000], ys: mnist.test.labels[:1000], keep_prob: 1})
        test_writer.add_summary(test, i)
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))

