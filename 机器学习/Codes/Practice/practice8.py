## 训练结果可视化，可以将优化的过程可视化出来,

"""
定义模型-->生成计算图-->创建会话-->开始训练
"""

import tensorflow as tf
import numpy as np
import time
import random


##---------tensorboard --logdir=H:\software\pycharm\Codes\practice\logs---------------##

# AF 是个激励函数，非线性方程
def add_layer(inputs, in_size, out_size, n_layer, AF=None):
    layer_name = 'layer%s' % n_layer
    # define layer name
    with tf.name_scope("layer"):
        # define weights name
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")  # 是个 in_size * out_size 矩阵
            tf.summary.histogram(layer_name + '/weights', Weights)  # tensorflow >= 0.12
        # define biases name
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")  # 一般是列表之类的（或者说是一维的向量[ ]），加上0.1后初始值不为0
            tf.summary.histogram(layer_name + '/biases', biases)  # Tensorflow >= 0.12
        # define Wx_plus_b name
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases  # 建立模型
        if (AF is None):
            outputs = Wx_plus_b
        else:
            outputs = AF(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)  # Tensorflow >= 0.12
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 定义区间以及增加维度、数据量大小，成为300*1的矩阵
noise = np.random.normal(0, 0.05, x_data.shape).astype(
    np.float32)  # 定义一个与 x_data shape相同的 noise（从高斯分布中获取样本） ，不然没办法与其相加，（平方shape不变）
y_data = np.square(x_data) - 0.5 + noise  # y_data由 x_data 运算而来，其shape是一样的

# 为输入输出定义 placeholder ，并添加到神经网络中
# define inputs name
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")  # 与上述的 x_data  shape类似形成 n*1 的矩阵，none即是代表没有限制，不定
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

## 典型的神经网络：输入层-隐藏层-输出层（3层网络）

# add hidden layer
hidden_layer1 = add_layer(xs, 1, 10, n_layer=1, AF=tf.nn.relu)  # 隐藏层 hidden layer

# add output layer
predition = add_layer(hidden_layer1, 10, 1, n_layer=2, AF=None)  # 输出层

# define loss name
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))  # 平方和求平均作为损失函数，其中的参数表示降维
    tf.summary.scalar('loss', loss)  # tensorflow >= 0.12

# define train name
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 采用梯度下降法优化求极值

# train_step=tf.train.MomentumOptimizer(0.1,0.5).minimize(loss)  #另外一种优化器

##在只有少量数据的情况下，优化器的改变对于训练速度的加快不是很明显

sess = tf.Session()  # 获得会话

merged = tf.summary.merge_all()  # tensorflow >= 0.12

# tf.train.SummaryWriter soon be deprecated, use following
from practice.util import logdirs

path = "H:\\software\\pycharm\\Codes\\practice\\"
logdirs(path + "log8s")
writer = tf.summary.FileWriter("log8s/", sess.graph)

init = tf.global_variables_initializer()  # 将所有的变量初始化

sess.run(init)

# 获取训练前的时间
timecurrent = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())) + '_' + str(random.randint(1000, 9999))
print(timecurrent)

for i in range(2000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})

        # 调用add_summary（）方法将训练过程数据保存在filewriter指定的文件中
        writer.add_summary(rs, i)  # 其中 i 代表坐标轴上的刻度

# 获取训练结束后的时间，与训练前的时间做对比
timecurrent = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())) + '_' + str(random.randint(1000, 9999))
print(timecurrent)
