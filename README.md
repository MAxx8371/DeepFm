# Tips:
## 1.Embedding Regulation
regularize specific embeddings whose id appear in current batch of data  
tf.trainable_variables获取GraphKeys.WEIGHTS,所以将id对应的embedding手动加入:  　
```
  tf.add_to_collection(tf.GraphKeys.WEIGHTS,embedding)  
  for var in tvars:  
    l2_loss = l2_loss + tf.nn.l2_loss(var)
```


## 数据处理
1. 连续数据处理：

2. 离散特征处理：

## DeepFm
1. FM算法：
category类的特征需要进行one-hot后输入FM， 例如颜色有[红,黄,蓝], 分别对应线性权重[w1,w2,w3], 一个红色的样本在颜色这个field的取值为[1,0,0], 与权重相乘的结果为w1. 这个过程就相当于embedding查找的过程.

分别开辟linear_weights和emb_weights
线性部分：不同于一般的FM乘以权重权重后直接求和，这里是concat然后过一层MLP

连续特征直接输入DNN，不经过FM？