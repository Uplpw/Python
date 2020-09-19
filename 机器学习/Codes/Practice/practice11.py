## 卷积神经网络CNN

"""
什么是卷积?
    对图像（不同的数据窗口数据）和滤波矩阵（一组固定的权重：因为每个神经元的多个权重固定，所以又可以看做一个恒定的滤波器filter）
    做内积（逐个元素相乘再求和）的操作就是所谓的『卷积』操作

图像上的卷积
   输入是一定区域大小(width*height)的数据，和滤波器filter（带着一组固定权重的神经元）做内积后等到新的二维数据。

   具体来说，左边是图像输入，中间部分就是滤波器filter（带着一组固定权重的神经元），不同的滤波器filter会得到不同的输出数据，比如颜色深浅、轮廓。
    相当于如果想提取图像的不同特征，则用不同的滤波器filter，提取想要的关于图像的特定信息：颜色深浅或轮廓。
"""

"""
定义模型-->生成计算图-->创建会话-->开始训练
"""
"""
用于构建ConvNet的层:　卷积层，池化层和完全连接层（与常规神经网络完全相同）

    但是用于CIFAR-10分类的简单ConvNet可以具有[INPUT-CONV-RELU-POOL-FC]架构。更详细地：
    
        １、INPUT [32x32x3]将保存图像的原始像素值，在这种情况下为宽度32，高度32并具有三个颜色通道R，G，B的图像。
        
        ２、CONV层将计算连接到输入中局部区域的神经元的输出，每个神经元计算它们的权重与它们连接到输入体积中的小区域之间的点积。
    如果我们决定使用12个滤镜，则可能会导致诸如[32x32x12]之类的体积。
    
        ３、RELU层将应用元素激活函数，例如m a x （0 ，x ）阈值为零。这使卷的大小保持不变（[32x32x12]）。
        
        ４、POOL层将沿空间尺寸（宽度，高度）执行下采样操作，从而产生诸如[16x16x12]的体积。
        
        ５、FC（即全连接）层将计算类别分数，从而得出大小为[1x1x10]的体积，其中10个数字中的每个数字都对应一个类别分数，例如CIFAR-10的10个类别。
    与普通的神经网络一样，顾名思义，该层中的每个神经元都将连接到上一卷中的所有数字。
    
    综上所述：

        在最简单的情况下，ConvNet体系结构是将图像量转换为输出量（例如保存类分数）的层列表。
        有几种不同类型的图层（例如，到目前为止，CONV / FC / RELU / POOL最受欢迎）
        每个图层接受输入3D体积，并通过可微分函数将其转换为输出3D体积
        每层可能有也可能没有参数（例如CONV / FC有，RELU / POOL没有）
        每个层可能有也可能没有其他超参数（例如CONV / FC / POOL有，RELU没有）
"""
import tensorflow as tf
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

#　define weight function
def weight_variable(shape, weights):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1, name="%s"%weights)
    return tf.Variable(initial)

# define bias function
def bias_variable(shape, bias):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32, name="%s"%bias)
    return tf.Variable(initial)

# 定义卷积神经网络层
"""
x 是图片的所有信息
W 是卷积层的 weight
"""
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
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

## 由于在 上述函数中输入的 x 参数，shape=[batch, height, width, channels],故需要处理一下 xs
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # print(x_image.shape)  # [n_samples, 28,28,1]

##---建立卷积层---##
## conv1 layer ##
with tf.name_scope("conv1_layer"):
    with tf.name_scope("conv1_Weights"):
        # patch 5x5, in size 1, out size 32, 原本数据是黑白，高度只有1，现在输出 32 是高度增加，和原理一样的操作
        W_conv1 = weight_variable([5, 5, 1, 32], weights="conv1_layer_weight")
    with tf.name_scope("conv1_bias"):
        b_conv1 = bias_variable([32], bias="conv1_layer_bias")
    # conv1 + 激励函数
    print("lpw: ", x_image.shape)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32, W_conv1输出的高度已经发生改变为 32
    # 池化 option
    h_pool1 = max_pool_2x2(h_conv1)            # output size 14x14x32， 池化是2*2，等于宽度和长度缩小1倍

## conv2 layer ##
with tf.name_scope("conv2_layer"):
    with tf.name_scope("conv2_Weights"):
        W_conv2 = weight_variable([5,5, 32, 64], weights="conv2_layer_weight") # patch 5x5, in size 32, out size 64 ，从第一次的 conv1 高度 32--->> 64 继续变高
    with tf.name_scope("conv2_bias"):
        b_conv2 = bias_variable([64], bias="conv2_layer_bias")
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
        W_fc1 = weight_variable([7*7*64, 1024], weights="FC1_layer_weight")
    with tf.name_scope("FC1_bias"):
        b_fc1 = bias_variable([1024], bias="FC1_layer_bias")

    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    ## overfitting option
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
# 最终输出是 0-9 的概率，只有 10 个
with tf.name_scope("FC2_layer"):
    with tf.name_scope("FC2_Weights"):
        W_fc2 = weight_variable([1024, 10], weights="FC2_layer_weight")
    with tf.name_scope("FC2_bias"):
        b_fc2 = bias_variable([10], bias="FC2_layer_bias")
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
with tf.name_scope("loss1"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))       # loss
    # 将 loss 曲线展现出来
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()


test_writer = tf.summary.FileWriter("log11s/", sess.graph)

# important step
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
save_path = saver.save(sess, "CNNSaver/cnnsave_net.ckpt")
print(save_path)

timecurrent = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
print(timecurrent)
# 开始训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # print("xs: ", batch_xs.shape)
    # print("ys: ", batch_ys.shape)
    _, loss=sess.run([train_step, prediction], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    # print(batch_ys)
    # print("loss:", loss)
    # print("shape:", prediction.shape)
    if i % 5 == 0:
        print(i, end=" ")
        timecurrent = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())) + '_' + str(random.randint(1000, 9999))
        print(timecurrent, end=" ")
        test=sess.run(merged, feed_dict={xs: mnist.test.images[:1000], ys: mnist.test.labels[:1000], keep_prob: 1})
        test_writer.add_summary(test, i)
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))

