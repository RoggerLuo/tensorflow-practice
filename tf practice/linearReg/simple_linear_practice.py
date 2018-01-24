import numpy as np
import tensorflow as tf

#准备数据  y = 2x + b , 
x = np.linspace(0,100,101)
#噪声
noisy = np.random.randn(*x.shape)*20
y = 2*x + noisy
print(y)


# 建立模型
pX = tf.placeholder('float',name = 'x')
pY = tf.placeholder('float',name = 'y')
W = tf.Variable(0.0,name='W')


Hypothesis  = tf.multiply(pX,W)
Cost = tf.pow((pY - Hypothesis),2)/len(x)/2


gdo = tf.train.GradientDescentOptimizer(0.01).minimize(Cost)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for  i in range(10):
    for (xi,yi) in zip(x,y):
        sess.run(gdo, feed_dict = {pX : xi,pY : yi})
        print(sess.run( W ))

print(sess.run( W ))

    