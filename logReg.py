import numpy as np
import tensorflow as tf
import pandas as pd
from  sklearn import preprocessing  

#准备数据  
df = pd.read_csv("./ex2data1.txt", header=None)
train_data = df.values
X = train_data[:,0:2]
y = train_data[:,2]
m = len(X)

# 标准化
scaler=preprocessing.StandardScaler().fit(X)  
X=scaler.transform(X)  

# 建立模型
pX = tf.placeholder(tf.float32,shape=(m,2))
pY = tf.placeholder(tf.float32,shape=(m,1))
W = tf.Variable(tf.random_normal([2, 1]),dtype=tf.float32) #tf.random_normal([2, 1]))
b = tf.Variable(tf.zeros([1, 1]),dtype=tf.float32)# //为什么这里是1,1  //tf.random_normal([1, 1])*0.1

# m,2 * 2,1 = m,1
Hypothesis  =  tf.sigmoid( tf.matmul(pX,W) + b)

# y:m,1  *  h:m,1  = 1,1
tY = tf.reshape(pY,[1,m])
CostRaw = -tf.matmul(tY,tf.log(Hypothesis)) - tf.matmul((1 - tY),tf.log(1 - Hypothesis))

Cost = tf.reduce_mean(tf.reduce_sum( CostRaw ,reduction_indices=[1]))/m #reduction_indices是指沿tensor的哪些维度求和
gdo = tf.train.GradientDescentOptimizer(0.02).minimize(Cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# X = X.reshape(-1,1,2)
y = y.reshape(m,1) 
costList = []
for  i in range(2000):
    # for (xi,yi) in zip(X,y):
    sess.run(gdo, feed_dict = {pX : X,pY : y})
    costList.append(sess.run( Cost, feed_dict ={pX : X,pY : y}))
    print(sess.run( Cost, feed_dict ={pX : X,pY : y}))

w = sess.run( W )
b = sess.run( tf.reduce_sum(b) )

import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(costList,'b.')

plt.subplot(1, 2, 2)
# transform之后的
x1 = X[:,0]#train_data[:, 0]
x2 = X[:,1]#train_data[:, 1]
y = train_data[:, -1:]

for x1p, x2p, yp in zip(x1, x2, y):
    if yp == 0:
        plt.scatter(x1p, x2p, marker='x', c='r')
    else:
        plt.scatter(x1p, x2p, marker='o', c='g')

x = np.linspace(20, 100, 2)
y = []
x=scaler.transform(x)  
# 这里的x,y都是 x来的，x是二维，混淆视听....
for i in x:
    y.append((i * -w[1] - b) / w[0])
plt.plot(x, y)# x1,x2

plt.show()
