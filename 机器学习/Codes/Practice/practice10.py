## 利用 Dropout 解决过拟合（verfitting）问题

"""
定义模型-->生成计算图-->创建会话-->开始训练
"""

# 首先加载数据集

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

##---------tensorboard --logdir=H:\software\pycharm\Codes\practice\classficationlogs---------------##

# sklearn中自带的手写数字数据集（digits数据集）
# 这个数据集中并没有图片，而是经过提取得到的手写数字特征和标记，就免去了我们的提取数据的麻烦，在实际的应用中是需要我们对图片中的数据进行提取的

# load data
digits = load_digits()
"""
digits.data：手写数字特征向量数据集，每一个元素都是一个64维的特征向量。

digits.target：特征向量对应的标记，每一个元素都是自然是0-9的数字。

digits.images：对应着data中的数据，每一个元素都是8*8的二维数组，其元素代表的是灰度值，转化为以为是便是特征向量
"""
# 训练集
X = digits.data
# 标记
y = digits.target

# 对标记进行二值化
y = LabelBinarizer().fit_transform(y)  # 将 y 数据集转为 0-9 对于索引位置的 0/1，与之前的 mnist 类似

# 将数据集分为训练集和测试集,比例是 7：3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


# AF 是个激励函数，非线性方程
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # here to dropout，
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    # 需要加上，不然会报错,需要和loss所用到的一起使用
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add hidden layer and output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss

# 将 loss 曲线展现出来
tf.summary.scalar('loss', cross_entropy)

# train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

# summary writer goes in here
from practice.util import logdirs

path = "H:\\software\\pycharm\\Codes\\practice\\"
logdirs(path + "log10s")
train_writer = tf.summary.FileWriter("log10s/train", sess.graph)
test_writer = tf.summary.FileWriter("log10s/test", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

# 开始训练
for i in range(500):
    # here to determine the keeping probability: 保证多少神经元不被 drop
    # in all, if keep_prob==1  overfitting, if keep_prob==0.5  not overfitting
    # train的时候才是dropout起作用的时候，test的时候不应该让dropout起作用
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        # record loss: train and test
        # test的时候不应该让dropout起作用
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})

        # 调用add_summary（）方法将训练过程数据保存在filewriter指定的文件中
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
