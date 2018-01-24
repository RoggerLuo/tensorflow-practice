# 变量，初始值为0，这个变量会被后面的optimizer所改变
# 加入了噪声： 以和trX同样的结构，生成正态分布的噪声
# sess.run(cost)需要为两个placeholder提供输入，
# cost是一个计算图，含义是：在当前的Variable(持久化的变量),和当下输入的xy，计算出cost


# 使用global_variables_initializer，init_all_variables depricated
# tf.mul 改成 tf.multiply


# 分开自己的重点和博客给别人看的重点,博客只要为自己服务
# 整理博客分类

import numpy as np
import tensorflow as tf

# 生成训练数据 + 噪声，下面为了拟合   Y = 2X   这个函数
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # y=2x，但是加入了噪声： 以和trX同样的结构，生成正态分布的噪声

# 构建计算图
X = tf.placeholder("float") #输入输出符号变量
Y = tf.placeholder("float")

# 定义模型
# 模型可以是线性函数，多项式，还可以是神经网络
def model(X, w):
    return tf.multiply(X, w) # 线性回归只需要调用乘法操作即可。

# 模型权重 W 用变量表示
w = tf.Variable(0.0, name="weights") # 变量，初始值为0，这个变量会被后面的optimizer所改变
y_model = model(X, w)

# 定义损失函数
cost = (tf.pow(Y-y_model, 2)) # 平方损失函数

# 构建优化器，最小化损失函数。
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 

# 构建会话
sess = tf.Session()

# 初始化所有的符号共享变量
init = tf.global_variables_initializer() 

# 运行会话
sess.run(init)

# 迭代训练
for i in range(100):
    for (x, y) in zip(trX, trY): 
        sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w))
# 打印权重w
print(sess.run(w))