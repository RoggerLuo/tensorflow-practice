def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1) # 返回axes，2行 3列 中的第1幅图
    plt.cla() #清除当前axes
    plt.plot(loss_list) 

    for batch_series_idx in range(5): #那5根葱...现在一根一根的看
        
        # 对list里所有predictions选择 第batch_series_idx个，
        # 选择第 batch_series_idx 跟葱
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        # 把最后那个2维的 处理下
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        # 每根葱对应一个axe
        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        # 坐标轴 0到15， 0到2
        plt.axis([0, truncated_backprop_length, 0, 2])
        
        #这个是啥, 横坐标嘛
        left_offset = range(truncated_backprop_length)
        
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)