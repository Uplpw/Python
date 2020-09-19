# 在Tensorflow中，定义了某字符串是变量，它才是变量，这一点是与Python所不同的。

# 定义语法： state = tf.Variable()

import tensorflow as tf

# 定义变量 state
state = tf.Variable(0, name='counter')

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)

# 如果你在Tensorflow中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后, 一定要定义 init = tf.initialize_all_variables().
# 到这里变量还是没有被激活，需要再在sess里, sess.run(init), 激活init这一步.

# 如果定义 Variable, 就一定要 initialize
init = tf.global_variables_initializer()  # 替换成这样就好

# 使用 Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

# 注意：直接print(state)不起作用！！一定要把sess的指针指向state再进行print才能得到想要的结果！