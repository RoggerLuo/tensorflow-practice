import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#准备数据  y = 2*X(1) + 10*X(2) + b  , 
x = np.random.rand(100)*10
x2 = np.random.rand(100)*1
#噪声
noisy = np.random.randn(*x.shape)*0.1
y = 2*x + noisy + 10*x2 + 95
# print(y)

# normalization省略了，也不知道语法
def fitModel(x,x2,y):
    # 建立模型
    pX = tf.placeholder(tf.float32,[1,2])
    pY = tf.placeholder(tf.float32,name = 'y')
    # W = tf.Variable(0.0,name='W')
    W = tf.Variable(tf.random_normal([2, 1]),dtype=tf.float32) #tf.random_normal([2, 1]))
    b = tf.Variable(tf.zeros([1, 1]) + 0.1,dtype=tf.float32)# //为什么这里是1,1

    Hypothesis  = tf.matmul(pX,W) + b
    Cost = tf.reduce_mean(tf.reduce_sum(tf.square(pY - Hypothesis),reduction_indices=[1])) #reduction_indices是指沿tensor的哪些维度求和
    # Cost = tf.pow((pY - Hypothesis),2)/len(x)/2
    # ,reduction_indices=[1]


    gdo = tf.train.GradientDescentOptimizer(0.001).minimize(Cost)

    sess = tf.Session()

    init = tf.global_variables_initializer()

    sess.run(init)

    # aaa =np.array([[8, 2]]) #np.random.randint(1,10,size=(1,2))
    # print(sess.run( Cost,feed_dict = {pX : aaa,pY : 2} ))

    x = np.array([x,x2])
    for i in range(2):
        x[i] = (x[i] - x[i].mean())/x[i].std()

    x = np.transpose(x)
    x = x.reshape(100,1,2)

    costList = []
    for  i in range(15):
        for (xi,yi) in zip(x,y):
            sess.run(gdo, feed_dict = {pX : xi,pY : yi})
            # print('Hypothesis:')
            # print(sess.run( Hypothesis,feed_dict = {pX : xi,pY : yi} ))

            # print('cost:')
            # print(sess.run( Cost,feed_dict = {pX : xi,pY : yi} ))
            costList.append(sess.run(Cost,feed_dict = {pX : xi,pY : yi}))
        # print('w')
        # print(sess.run( W ))
        # print('b')
        # print(sess.run( b ))
        # print('---割---')
    print(len(costList))
    # plt.ion()
    return costList

    print(sess.run( W ))
    print(sess.run( b ))



#准备数据  y = 2*X(1) + 10*X(2) + b  , 
x = np.random.rand(100)*10
x2 = np.random.rand(100)*1
#噪声
noisy = np.random.randn(*x.shape)*0.1
y = 2*x + noisy + 10*x2 + 95
# print(y)
cost1 = fitModel(x,x2,y)

xx = np.random.rand(100)*10
xx2 = np.random.rand(100)*10
#噪声
noisy = np.random.randn(*x.shape)*0.1
yy = 2*x + noisy + 10*x2 + 95
# print(y)
cost2 = fitModel(xx,xx2,yy)



xxx = np.random.rand(100)*10
xxx2 = np.random.rand(100)*10
#噪声
noisy = np.random.randn(*x.shape)*10
yyy = 2*x + noisy + 10*x2 + 95
# print(y)
cost3 = fitModel(xxx,xxx2,yyy)



plt.figure(figsize=(20,7))
plt.subplot(1, 3, 1)
plt.plot(cost1,'b.')
plt.subplot(1, 3, 2)
plt.plot(cost2,'b.')
plt.subplot(1, 3, 3)
plt.plot(cost3,'b.')

# plt.ioff()
plt.show()

