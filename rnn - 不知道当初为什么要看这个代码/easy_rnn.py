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

num_layers = 3

def generateData():
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step) 
    y[0:echo_step] = 0
    x = x.reshape((batch_size, -1))  
    y = y.reshape((batch_size, -1))
    return (x, y)


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

# replace 2
# cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
# hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
# init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)


# 变化多的第一部分
# 这一部分是准备 init_state的placeholder
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size]) # 3,2,5,4

state_per_layer_list = tf.unstack(init_state, axis=0) #竖着切？  那不就是把第一层次的每个item放到一个list里面,一共3个，0 1 2
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)
# 一共三个init_state，每个里面都有cell[0]和hidden[1]



W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
# inputs_series = tf.unstack(batchX_placeholder, axis=1) 
labels_series = tf.unstack(batchY_placeholder, axis=1)
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1) # (5,1)


# Forward pass
# current_state = init_state
# states_series = [] 


# for current_input in inputs_series: 
#     current_input = tf.reshape(current_input, [batch_size, 1])
#     input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
#     next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b) 
#     states_series.append(next_state)
#     current_state = next_state

# hidden W and b，W只要一个state_size就行，b也是，所以cell只有这一个参数
# init_state (5,4)
# inputs_series [(5,1)]
# 把inputs_series变成states_series， 中间用了cell当作？胶水？总之把(state_size+1)映射成state_size

# replace3
# cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
# states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, initial_state = init_state)


# 变化多的第二部分
# 这一部分是把inputPlaceholder映射到 states_series(placeholder), 
# 也用上了 init_state的placeholder，之前定义的
def lstm_cell():
    return tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
stacked_lstm_cell = [lstm_cell() for _ in range(num_layers)]

cell = tf.nn.rnn_cell.MultiRNNCell(stacked_lstm_cell, state_is_tuple=True)

states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, initial_state=rnn_tuple_state)
#tf.contrib.rnn.static_rnn(cell, inputs_series, initial_state=rnn_tuple_state)



logits_series = [tf.matmul(state, W2) + b2 for state in states_series] # Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
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
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()


        # 变化多的第三部分
        # 真正的 init_state的 初始数据 ，喂给placeholder的
        # _current_cell_state = np.zeros((batch_size, state_size))
        # _current_hidden_state = np.zeros((batch_size, state_size))   
        _current_state = np.zeros((num_layers, 2, batch_size, state_size))

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
                    # cell_state: _current_cell_state,
                    # hidden_state: _current_hidden_state
                    init_state:_current_state
                })
            # _current_cell_state, _current_hidden_state = _current_state

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()