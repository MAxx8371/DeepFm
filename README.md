# Tips:
## 1.Embedding Regulation
regularize specific embeddings whose id appear in current batch of data  
tf.trainable_variables获取GraphKeys.WEIGHTS,所以将id对应的embedding手动加入:  　
```
  tf.add_to_collection(tf.GraphKeys.WEIGHTS,embedding)  
  for var in tvars:  
    l2_loss = l2_loss + tf.nn.l2_loss(var)
```
