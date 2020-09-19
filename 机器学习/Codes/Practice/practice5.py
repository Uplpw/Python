## 添加子层 layer()

import tensorflow as tf
import numpy as np

# AF 是个激励函数，非线性方程
def add_layer(inputs, in_size, out_size, AF=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size])) #是个 in_size * out_size 矩阵
    biases=tf.Variable(tf.zeros([1, out_size])+0.1) # 一般是列表之类的（或者说是一维的向量[ ]），加上0.1后初始值不为0
    Wx_plus_b=tf.matmul(inputs, Weights)+biases   # 建立模型
    if(AF is None):

        outputs=Wx_plus_b
    else:
        outputs=AF(Wx_plus_b)
    return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis] #定义区间以及增加维度、数据量大小，成为300*1的矩阵
noise=np.random.normal(0,0.05,x_data.shape) #定义一个与 x_data shape相同的 noise（从高斯分布中获取样本） ，不然没办法与其相加，（平方shape不变）
y_data=np.square(x_data)-0.5+noise  # y_data由 x_data 运算而来，其shape是一样的

xs=tf.placeholder(tf.float32, [None, 1])  #与上述的 x_data  shape类似形成 n*1 的矩阵，none即是代表没有限制，不定
ys=tf.placeholder(tf.float32, [None, 1])

## 典型的神经网络：输入层-隐藏层-输出层（3层网络）

layer1=add_layer(xs, 1, 10, AF=tf.nn.relu) #隐藏层

predition=add_layer(layer1, 10, 1, AF=None)  #输出层

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition), reduction_indices=[1])) #平方和求平均作为损失函数，其中的参数表示降维

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)  #采用梯度下降法优化求极值

init=tf.initialize_all_variables() #将所有的变量初始化

sess=tf.Session() #获得会话

sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50==0:
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
