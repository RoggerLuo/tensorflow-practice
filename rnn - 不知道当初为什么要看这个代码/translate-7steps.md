### 生成数据
现在我们生成训练数据，输入的数据是一个二进制的向量。
输出数据可以看成是输入数据向右偏移了echo_step个位置的“回音”， 

``` python
def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)
```
**np.random.choie( 2, `total_series_length`, p=[0.5,0.5] )**  
第一个参数：从np.arrange(2)，即`[0,1]`这个数组中，  
第二个参数：随机生成一个新的，长度为`total_series_length`的数组,  
第三个参数：为0的概率和为1的概率都为0.5(均等分布)  
	
**y = np.roll( x, echo_step )**  
把数组x向右滚动echo_step个位置，右边超出的元素放回最左边

**y[ 0 : echo_step ] = 0**  
把数组y从下标从`0`到`echo_step`的项赋值为0

**x.reshape( ( batch_size, -1 ) )**  
把一位数组x重新塑造成一个行数为`batch_size`的二维数组，
column的长度根据数组长度推断(计算)得出  

参数**( batch_size, -1 )**是新数组的结构信息

* * *
需要注意的是，在把一维数据变形为矩阵过程中，`batch_size`决定了矩阵的行数。  

我们通过不断地向(损失函数相对于神经元权重W的)梯度方向靠近来训练神经网络，
并且只用整个数据的一小部分子集来训练，这个方法也叫做`mini-batch`。  
这么做的原因是：`请参考这一篇文章`。

数据结构的变形的作用是：把数据集整合放进一个矩阵，之后会把这个矩阵分成一个一个的子集，用于之前说的`mini-batch`方法。

### 建构计算图(graph)
第一次看到这个名词其实很难形象的理解，于是我找来了一些注解。
	
	图(graph)是一种比较松散的数据结构。  
	它有一些节点(vertice)，在某些节点之间，由边(edge)相连。  
	节点的概念在树中也出现过，我们通常在节点中储存数据。边表示两个节点之间的存在关系。  
	在树中，我们用边来表示子节点和父节点的归属关系。树是一种特殊的图，但限制性更强一些。
	／／图片
	这样的一种数据结构是很常见的。  
	比如计算机网络，就是由许多节点(计算机或者路由器)以及节点之间的边(网线)构成的。  
	城市的道路系统，也是由节点(路口)和边(道路)构成的图。  
	地铁系统也可以理解为图，地铁站可以认为是节点。  
	基于图有许多经典的算法，比如求图中两个节点的最短路径，求最小伸展树等。
	／／图片

TensorFlow工作的时候，会首先建立一个计算图，它决定之后会进行什么样操作。
一般来说，计算图的输入和输出都是多位数组，也被称作`tensors`。  
计算图，或者计算图的一部分，可以在一次会话(session)中多次反复使用，  
这些操作可以在CPU、GPU,甚至在远程服务器上来完成。

### 变量和占位符
变量和占位符是TensorFlow两种基本的数据结构。  

	tf.Variable  
	tf.placeholder  
每一轮运行的数据集都会送到`占位符`那里，这些`占位符`作为计算图的开始节点。  
`占位符`里也提供RNN的状态，用来保存之前一次运行的输出结果。

```python
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])
```

网络的权值和偏置都声明称TensorFlow变量，这样可以让他们持久化，在每一轮训练中都更新自己值。


```python
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)
```

挺下图展示了输入数据，虚线框里的是当前训练数据集`batchX_placeholder`。  
我们之后会看到，“训练集窗口”每一轮向右滑了`truncated_backprop_length`步。  
下图的案例中，  
`batch_size = 3`训练数据集的大小为3,  
`truncated_backprop_length = 3`,  
`total_series_length =36`系列总长度为36,
请注意这些数据只是为了视觉展示目的，在代码中的参数和这个是不一样的。这个系列数据的部分数据点显示了序号。  

//图片

### 解包(unpacking)
现在是时候建构计算图了，模拟真实的RNN计算，首先我们把训练集数据分隔开，放到临近的位置上。  

```python
# Unpack columns
inputs_series = tf.unpack(batchX_placeholder, axis=1)
labels_series = tf.unpack(batchY_placeholder, axis=1)
```
就像你在下图中看到的，把训练子集中列放置到Python的list中。  
RNN将会同时对整个数据集中的几个不同部分进行训练；从4到6，从16到18，28到30。  
之所以用"plural_series"(复合序列)作为变量的名字，是为了强调我们的变量是一个多入口的序列。  
 
//图  

事实上我们的训练时在三个地方同时发生的，执行向前传播的时候，需要我们保存三份状态的实例。  
我们已经考虑到这种情况了，所以初始状态的`placeholder`有`batch_size`行。



### 向前传递
接下来我们建构真正执行RNN计算的那一部分。

```python 
# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat(1, [current_input, current_state])  # Increasing number of columns

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state    

```












### 运行一次训练会话(training session)

是时候集中注意力，开始训练我们的网络了，在TensorFlow中graph是在session中执行的。
每一轮 ( )都会产生新数据,(并不是用常规的办法来做到这一点，但是因为所有的东西都是可预见的，所以在这个例子里面这么做是有效的，)
