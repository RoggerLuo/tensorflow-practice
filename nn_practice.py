import numpy as np
import tensorflow as tf

#mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X = mnist.train.images[0:5000,:]
y = mnist.train.labels[0:5000,:]
Xt = mnist.test.images[0:1000,:]
yt = mnist.test.labels[0:1000,:]


# 输入的
px = tf.placeholder(tf.float32,[None,784])
py = tf.placeholder(tf.float32,[None,10])

# 要训练的
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
hyp = tf.nn.softmax(tf.matmul(px,W)+b)
# cross entropy
cost = - tf.reduce_sum(py*tf.log(hyp))
# 训练
train_once = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# start
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
costList = []
for i in range(50):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_once, feed_dict = {px: batch_xs, py: batch_ys})
    costList.append(sess.run( cost, feed_dict ={px: batch_xs, py: batch_ys}))
    print(sess.run(b))

import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(costList,'b-')
plt.show()

# 计算准确率，是拿输出的结果比对一下
corre_boolean = tf.equal(tf.argmax(hyp,1),tf.argmax(py,1))
accu = tf.reduce_mean(tf.cast(corre_boolean,'float'))
print(sess.run(accu, feed_dict={px: Xt, py: yt}))

# 显示图片
# import matplotlib.pyplot as plt
# plt.figure(figsize=(16, 8), dpi=80)
# displayX = X[5:10].reshape(5,28,28)
# for ind in range(len(displayX)):
#     axes = plt.subplot(2,3,ind+1)
#     pic = displayX[ind]
#     for rowInd in range(len(pic)):
#         oneRow = pic[rowInd]
#         for eachPixelInd in range(len(oneRow)):
#             value = oneRow[eachPixelInd]
#             plt.plot(eachPixelInd,rowInd,marker='*',c=(value,value,value))
# plt.show()

