# 整个案例的感觉就是增加了一个state Variable
# 然后对这个Variable拥有的权重进行训练
# 所以看成是 ‘增加门’，还有个 ‘忘记门’
# 真是艰辛啊，一路看下来，至少花了2个工作日

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    # 1.用不用np.array包裹都没差，
    # 2.返回一个 total_series_length 长度的数组, 从 0 1 中随机选出来的数组
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5])) 
    
    # 整体向右移动echo_step个数
    y = np.roll(x, echo_step) 
    
    # 从后面被挤到前面的项设置为0
    y[0:echo_step] = 0

    # x 的length要能被batch_size整除才行, x是一维数组
    # -1 表示自动计算出column的数量,
    x = x.reshape((batch_size, -1))  
    y = y.reshape((batch_size, -1))
    return (x, y)

# batch_size行, 每行truncated_backprop_length列，可能是为了让每一个批次不要太大
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

# what fuck is that, init_state:5x4 ??? 好吧，跟W一样
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# 为什么是state_size+1, 数据结构是 5x4
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

# 为什么有W2,b2
# 结构是 4 x 2 ??? , num_classes = 2,
W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1) #按列解包
labels_series = tf.unstack(batchY_placeholder, axis=1) #按列解包
# batchX_placeholder是5x15的一个二维数组
# 按列解包，15x5，第一个15是list，有15个项， 每一项是一个长度为5的array，不知道会有什么影响
# 15 5

# Forward pass
# current_state: 5x4 
current_state = init_state
# attention, 这里又多了一个states_series
states_series = [] 

# 这里只执行一次，逻辑上好难理解
# 把placeholder变形了以后(长度是固定的15长度的list)，
# 然后分别对这15个placeholder进行一些操作
# 每次输入，把新的输入和老的state合并，然后乘以W一套操作，得出next_state
for current_input in inputs_series: 
    # 对placeholder的每一项"current_input"进行reshape
    # current_input是15list中的一项，这一项中是5(batch_size)length的array
    # reshape以后，这个一维的shape为(batch_size,)的数组，变成了一个二维的，shape为(batch_size,1)的数组
    # 5x1
    current_input = tf.reshape(current_input, [batch_size, 1])
    # 把这一项和当前状态合并，wait这里有问题，1表示列 
    # current_state: 5x4 
    input_and_state_concatenated = tf.concat(1, [current_input, current_state])  # Increasing number of columns
    # 把之前的state加上新的input，所以是state+1的size
    # 之前的current_state不是乘了W很多很多次...
    # 叠加的乘W，
    # input_and_state_concatenated的row是不断增加的，但是column没有增加，
    # 卧槽，这里把 state_size+1 变成了==> state_size,所以next_state永远是 1x4
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    
    # 所以states_series 是一直增加的吗，
    # states_series就有init_state(4)+len(inputs_series)个 
    states_series.append(next_state)
    current_state = next_state


# 所以这一段循环
# 把placeholderX变成了states_series，加入了W和b
# 而且states_series 固定了，有 4(init_state)+len(inputs_series)个 ，inputs_series也是固定的
# 总之就是，把 placeholderX --映射成--> states_series， 结构式固定的
# 但是每次placeholderX的输入不同，所以这个结构里填的值不同


# W2:4x2 
# 得到一个逻辑分类，用[ 0 1 0 1 ]的形式表示，因为只有两类，所有只有两位长度
# logits_series里的每一个都是一个两位的数组 
# logits的标准，分几类就是几，num_classes是几，就是几位的数组
# tf.nn.softmax把一个数组，按照 总和为1，数值大小的比例，来调整数组里的每一项

# softmax_cross_entropy_with_logits
# loss = sess.run(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= y_)) 得到一个损失数组
# tf.reduce_mean(loss)把这个损失数组平均了一下

# tf.nn.sparse_softmax_cross_entropy_with_logits
# A common use case is to have logits of shape [batch_size, num_classes] and labels of shape [batch_size]
# labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result) and dtype int32 or int64. Each entry in labels must be an index in [0, num_classes).
# since it performs a softmax on logits internally for efficiency. Do not call this op with the output of softmax, as it will produce incorrect results.

# states_series就是 输入的x与之前有的state进行一些运算之后得到的最终形式，
# 等于说state_series才是真正的 input, W2和b2是基于这个input的权重
# 再所以， 有两个部分的w和b,相当于某种形式的两层神经元？
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] # Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            # 看来stride为0, 没有重叠
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
            # 保持5行不变，每次截取truncated_backprop_length这个长度
            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()