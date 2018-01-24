 TensorFlow搭建RNN(2/7)  使用TensorFlow的RNN API  
这一篇文章是[TensorFlow搭建RNN(1/7) 简单案例](http://blog.csdn.net/sinat_24070543/article/details/75113014)的后续文章，  
前一篇文章里，我们从零建立了一个RNN，手动建立计算图，现在我们用TensorFlow原生API来简化我们的代码。  
### 计算图的简单创建
``` python
inputs_series = tf.unstack(batchX_placeholder, axis=1) 
labels_series = tf.unstack(batchY_placeholder, axis=1) 

for current_input in inputs_series: 
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b) 
    states_series.append(next_state)
    current_state = next_state

```
把之前的代码(上面)换成下面的，

``` python
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, initial_state = init_state)
```

还有，你可以把之前的权重和偏置矩阵W和b的声明部分也移除了，  
这些都隐藏在RNN的api里面的了。

看看这次的变化
### cell 
``` python
cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
```
看看之前W、b的定义：

``` python
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)
```
观察，W和b都只有一个可变参数，就是`state_size`,  
现在把 W和b放进了cell里面，传入`state_size`
### x_inputs
``` python
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
```
用`split`代替了`unstack`， `split`沿着axis=1将tensor分解成更小的tensors，  
这里inputs_series的shape是`(5,1)`  

而在之前的代码中,unstack把最后一个维度移除了，shape为`(5,)`,  
所以我们才又在for循环中reshape一次，把`(5,)`转成`(5,1)`

### tf.contrib.rnn.static_rnn 
原文`tf.nn.rnn`，现在替换为`tf.contrib.rnn.static_rnn `  
代替了for循环，  
把inputs和cell结合生成了states的序列，  
返回的数据也和之前的一样：`states_series, current_state`

## 下一步
下一篇我们将用LSTM的架构来完善RNN。
虽然这个案例比较简单，但我们的目的是学习TensorFlow。


原文来自medium:  
https://medium.com/@erikhallstrm/tensorflow-rnn-api-2bb31821b185  

本文根据tensorflow 1.2的api修改了代码