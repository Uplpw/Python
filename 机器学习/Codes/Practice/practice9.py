## 使用TensorFlow解决Classification（分类）问题-----手写数字

"""
定义模型-->生成计算图-->创建会话-->开始训练
"""

## MNIST 数据
# 首先准备数据（MNIST库）
"""
MNIST库是手写体数字库，数据中包含55000张训练图片，每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from practice.util import logdirs
import time
import random

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# AF 是个激励函数，非线性方程
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None,):
    layer="layer%s"%n_layer
    # add one more layer and return the output of this layer
    with tf.name_scope("%s"%layer):
        with tf.name_scope("Weigths"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name="W")
            tf.summary.histogram("W", Weights)
        with tf.name_scope("bias"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
            tf.summary.histogram("B", biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b,)
        tf.summary.histogram(layer + '/outputs', outputs)
        return outputs

# 计算网络的准确率
def compute_accuracy(v_xs, v_ys):
    global prediction
    # 使用测试数据集进行预测结果，预测值是 1* 10 的， 每个元素是一个概率，值越大代表可能性越大，最大值的位置即预测的 数字
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})

    # 使用测试数据集的准确结果，与真实值对比（有正确也有错误）
    # tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引
    """
        axis=0时比较每一列的元素，将每一列最大元素所在的索引记录下来，最后输出每一列最大元素所在的索引数组。
        axis=1的时候，将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组。
        由于 y_pre 和 v_ys 是1*10的，所以axis的值是 1，索引每行的最大值位置
    """
    # tf.argmax(y_pre,1) 即函数值为 1 时也就是元素为 1 所在的位置
    """
        tf.equal()用法：equal(x, y, name=None) 就是判断，x, y 是不是相等，它的判断方法不是整体判断，而是逐个元素进行判断，如果相等就是True，不相等，就是False。

        由于是逐个元素判断，所以x，y 的维度要一致。
    """
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))

    # 计算准确率，即计算 correct_prediction 中有多少正确以及多少错误
    # tf.cast强制类型转换，将 correct_prediction 布尔型的转为 tf.float32
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #最后求平均

    # sess.run()
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


## 搭建网络
# define placeholder for inputs to network
# 输入的是多张手写体照片，分辨率是28×28，应该是28×28=784个像素数据
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 784], name="x_input") # 28x28
    # 每张图片都表示一个数字，所以我们的输出是数字0到9，共10类。
    ys = tf.placeholder(tf.float32, [None, 10], name="y_input")

# add ouput layer
# 调用add_layer函数搭建一个最简单的训练网络结构，只有输入层和输出层。其中的输入输出参数即是我们的输入 784 和 10
prediction = add_layer(xs, 784, 10, n_layer=1, activation_function=tf.nn.softmax)

# 损失函数，其中 cross_entropy 是专用处理 分类 的损失函数
with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
    tf.summary.scalar('loss', cross_entropy)

# 训练， 采用梯度下降法优化求极值
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

## very important，建立会话并初始化所有的变量以及进行计算
sess = tf.Session()
merged = tf.summary.merge_all()
# tf.train.SummaryWriter soon be deprecated, use following
path="H:\\software\\pycharm\\Codes\\practice\\"
logdirs(path+"log9s")
writer = tf.summary.FileWriter("log9s/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

# 获取训练前的时间
timecurrent = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
print(timecurrent)
## 开始训练
for i in range(2000):
    ## 整个数据集分为 training data and test data, 防止人为因素的影响
    # 从 训练数据集 中获取批量数据100，否则每次训练数据量很大，比较耗费时间
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        # 计算准确率
        print(compute_accuracy(mnist.test.images, mnist.test.labels)) # 使用 测试数据集 进行计算
        rs=sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys})
        writer.add_summary(rs, i)

# 获取训练结束后的时间，与训练前的时间做对比
timecurrent = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
print(timecurrent)
