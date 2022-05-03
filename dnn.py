import tensorflow as tf
from tensorflow.python.layers import normalization
        
class deepFM:
  def __init__(self):
      self._weights = {}
      
  def add_weights(self,params):
    emb_dim = params['embedding_dim']
    for vocab_name, vocab_size in params['vocab_sizes'].items():
      linear_weight = tf.get_variable(
          name='{}_linear_weight'.format(vocab_name),
          shape=[vocab_size,1],
          initializer=tf.glorot_normal_initializer(),
          dtype=tf.float32
      )
      
      emb_weight = tf.get_variable(
        name='{}_emb_weight'.format(vocab_name),
        shape = [vocab_size, emb_dim],
        initializer=tf.glorot_normal_initializer(),
        dtype=tf.float32
      )
      self._weights[vocab_name] = (linear_weight,emb_weight)
          
  def get_linear_weights(self,vocab_name):
    return self._weights[vocab_name][0]
  
  def get_emb_weights(self,vocab_name):
    return self._weights[vocab_name][1]
          
  def output_logits_from_linear(self, features, params):
      field2vocab_mapping = params['field_vocab_mapping']
      combiner = params.get('multi_embed_combiner','sum')
      
      fields_outputs = []
      
      for fieldname, vocabname in field2vocab_mapping.items():    #items():以列表返回可遍历的(键, 值) 元组数组
          sp_ids = features[fieldname+"_ids"]
          sp_values = features[fieldname+"_values"]
          
          linear_weights = self.get_linear_weights(vocab_name=vocabname)
          
          #根据SparseTensor,按照每一行取出embedding,并聚合(默认为mean)
          output = tf.nn.safe_embedding_lookup_sparse(linear_weights, sp_ids, sp_values,
                                                      default_id=0,
                                                      combiner=combiner,
                                                      name = '{}_linear_output'.format(fieldname))
          
          fields_outputs.append(output)
          
      whole_linear_output = tf.concat(fields_outputs,axis=1)
      
      tf.logging.info("linear output, shape={}".format(whole_linear_output.shape))
      logits = tf.layers.dense(whole_linear_output,units=1, use_bias=True, activation=None)
      
      return logits
  
  def output_logits_from_bi_interaction(self,features,params):
      field2vocab_mapping = params['field_vocab_mapping']
      
      combiner = params.get('multi_embed_combiner','sum')
      
      #对于多值的field,使用聚合的方法将为field产生一个embedding
      fields_embeddings = []
      fields_squared_embedding = []
      
      for fieldname, vocabname in field2vocab_mapping.items():
        sp_ids = features[fieldname + "_ids"]
        sp_values = features[fieldname + "_values"]
        
        embed_weights = self.get_emb_weights(vocabname)
        
        embedding = tf.nn.safe_embedding_lookup_sparse(embed_weights,sp_ids,
                                                        combiner=combiner,
                                                        name='{}_embedding'.format(fieldname),
                                                        default_id=0)
        fields_embeddings.append(embedding)
        
        squared_emb_weights = tf.square(embed_weights)
        
        squared_embedding = tf.nn.safe_embedding_lookup_sparse(squared_emb_weights,sp_ids,
                                                              combiner=combiner,
                                                              name='{}_squared_embedding'.format(fieldname),
                                                              default_id=0)
        
        fields_squared_embedding.append(squared_embedding)
          
      sum_embedding_then_square = tf.square(tf.add_n(fields_embeddings))
      square_embedding_then_sum = tf.add_n(fields_squared_embedding)
      
      bi_iteraction = 0.5 * (sum_embedding_then_square - square_embedding_then_sum)
      tf.logging.info("bi-interaction, shape={}".format(bi_iteraction.shape))
      
      logits = tf.layers.dense(bi_iteraction,units=1,use_bias=True,activation=None)
      
      return logits, fields_embeddings
  
  #连续特征直接输入DNN,不经过FM
  #需要将连续特征离散化
  #这里不加self会报错，为什么？
  def output_logits_from_dnn(self, fields_embeddings, params, is_training=True):
    dropout_rate = params['dropout_rate']
    do_batch_norm = params['batch_norm']
    
    X = tf.concat(fields_embeddings, axis=1)
    tf.logging.info("initial input tp DNN, shape={}".format(X.shape))
    
    for idx, n_units in enumerate(params['hidden_units'],start=1):
      X = tf.layers.dense(X,units=n_units,activation=tf.nn.relu)
      tf.logging.info("layer[{}] output shape={}".format(idx,X.shape))
      
      X = tf.layers.dropout(inputs=X, rate=dropout_rate, training=is_training)
      if is_training:
          tf.logging.info("layer[{}] dropout {}".format(idx,dropout_rate))
            
    if do_batch_norm:
      # BatchNormalization的调用、参数，是从DNNLinearCombinedClassifier源码中拷贝过来的
      batch_norm_layer = normalization.BatchNormalization(momentum=0.999, trainable=True,
                                                          name='batchnorm_{}'.format(idx))
      X = batch_norm_layer(X, training=is_training)

      if is_training:
          tf.logging.info("layer[{}] batch-normalize".format(idx))
    
    return tf.layers.dense(X, units=1, use_bias=True, activation=None)