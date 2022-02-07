from re import A
import tensorflow as tf
from deepfm import deepFm
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def to_sparse_input_and_drop_ignore_values(input_tensor, ignore_value=None):
    """Converts a `Tensor` to a `SparseTensor`, dropping ignore_value cells.
    If `input_tensor` is already a `SparseTensor`, just return it.
    Args:
      input_tensor: A string or integer `Tensor`.
      ignore_value: Entries in `dense_tensor` equal to this value will be
        absent from the resulting `SparseTensor`. If `None`, default value of
        `dense_tensor`'s dtype will be used ('' for `str`, -1 for `int`).
    Returns:
      A `SparseTensor` with the same shape as `input_tensor`.
    Raises:
      ValueError: when `input_tensor`'s rank is `None`.
    """
    input_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
        input_tensor)
    if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
        return input_tensor
    with ops.name_scope(None, 'to_sparse_input', (input_tensor, ignore_value,)):
        if ignore_value is None:
            if input_tensor.dtype == dtypes.string:
                # Exception due to TF strings are converted to numpy objects by default.
                ignore_value = ''
            elif input_tensor.dtype.is_integer:
                ignore_value = -1  # -1 has a special meaning of missing feature
            else:
                # NOTE: `as_numpy_dtype` is a property, so with the parentheses this is
                # constructing a new numpy object of the given type, which yields the
                # default value for that type.
                ignore_value = input_tensor.dtype.as_numpy_dtype()
        ignore_value = math_ops.cast(
            ignore_value, input_tensor.dtype, name='ignore_value')
        indices = array_ops.where(
            math_ops.not_equal(input_tensor, ignore_value), name='indices')
        return sparse_tensor_lib.SparseTensor(
            indices=indices,
            values=array_ops.gather_nd(input_tensor, indices, name='values'),
            dense_shape=array_ops.shape(
                input_tensor, out_type=dtypes.int64, name='dense_shape'))

COLUMNS_MAX_TOKENS = []
for i in range(1,14):
    COLUMNS_MAX_TOKENS.append(('I{}'.format(i),1))
COLUMNS_MAX_TOKENS.append(('multi_feats',3))
for i in range(4,24):
    COLUMNS_MAX_TOKENS.append(('C{}'.format(i),1))   

DEFAULT_VALUES = [[0]] + list([''] for i in range(13)) + list([''] for i in range(24))

# tmp = "1:2,2:3,3:4"
# kvpair = tf.string_split([tmp],',').values
# kvpair = tf.string_split(kvpair,':')
# kvpair = tf.reshape(kvpair.values, kvpair.dense_shape)
# feat_ids, feat_vals = tf.split(kvpair, num_or_size_splits=2, axis=1)
# print(feat_ids)

def _decode_csv(line):
  columns = tf.decode_csv(line,record_defaults=DEFAULT_VALUES, field_delim='\t')
  y = columns[0]
  
  feat_columns = dict(zip((t[0] for t in COLUMNS_MAX_TOKENS),columns[1:]))
  X = {}
  for colname, max_tokens in COLUMNS_MAX_TOKENS:
      kvpairs = tf.string_split([feat_columns[colname]],',').values[:max_tokens]
      
      kvpairs = tf.string_split(kvpairs,':')
      
      kvpairs = tf.reshape(kvpairs.values, kvpairs.dense_shape)
      
      feats_ids, feats_vals = tf.split(kvpairs,num_or_size_splits=2,axis=1)
      feats_ids = tf.string_to_number(feats_ids,out_type=tf.int32)
      feats_vals = tf.string_to_number(feats_vals,out_type=tf.float32)
      
      X[colname + "_ids"] = tf.reshape(feats_ids,shape=[-1])
      X[colname + "_values"] = tf.reshape(feats_vals,shape=[-1])
      
  return  X, y

def input_fn(data_file,n_repeat,batch_size,batches_per_shuffle):
  # ----------- prepare padding
  pad_shapes = {}
  pad_values = {}
  for c, max_tokens in COLUMNS_MAX_TOKENS:
    pad_shapes[c + "_ids"] = tf.TensorShape([max_tokens])
    pad_shapes[c + "_values"] = tf.TensorShape([max_tokens])

    pad_values[c + "_ids"] = -1  # 0 is still valid token-id, -1 for padding
    pad_values[c + "_values"] = 0.0

  # no need to pad labels
  pad_shapes = (pad_shapes, tf.TensorShape([]))
  pad_values = (pad_values, 0)

  # ----------- define reading ops
  dataset = tf.data.TextLineDataset(data_file).skip(1)  # skip the header
  dataset = dataset.map(_decode_csv, num_parallel_calls=4)

  if batches_per_shuffle > 0:
      dataset = dataset.shuffle(batches_per_shuffle * batch_size)

  dataset = dataset.repeat(n_repeat)
  dataset = dataset.padded_batch(batch_size=batch_size,
                                  padded_shapes=pad_shapes,
                                  padding_values=pad_values)

  iterator = dataset.make_one_shot_iterator()
  dense_Xs, ys = iterator.get_next()

  # ----------- convert dense to sparse
  sparse_Xs = {}
  for c, _ in COLUMNS_MAX_TOKENS:
      for suffix in ["ids", "values"]:
          k = "{}_{}".format(c, suffix)
          sparse_Xs[k] = to_sparse_input_and_drop_ignore_values(dense_Xs[k])

  # ----------- return
  return sparse_Xs, ys

def model_fn(features,labels,mode,params):
    for featname, featvalues in features.items():
        if not isinstance(featvalues, tf.SparseTensor):
            raise TypeError("feature[{}] isn't SparseTensor".format(featname))
        
    model = deepFm()
    model.add_weights(params)
    
    linear_logits = model.output_logits_from_linear(features,params)
    
    bi_interact_logits, fields_embeddings = model.output_logits_from_bi_interaction(features,params)
    
    dnn_logits = model.output_logits_from_dnn(fields_embeddings,params,(mode==tf.estimator.ModeKeys.TRAIN))

    bias = tf.get_variable(name='general_bias',shape=1,initializer=tf.constant_initializer(0.0))
    
    logits = linear_logits + bi_interact_logits + dnn_logits
    logits = tf.add(logits,bias)
    
    logits = tf.reshape(logits,shape=[-1])
    
    probabilities = tf.sigmoid(logits)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={'probabilities' : probabilities}
      )
        
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.cast(labels, tf.float32)))
    
    eval_metric_ops = {'auc': tf.metrics.auc(labels, probabilities)}
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode = mode,
          loss = loss,
          eval_metric_ops=eval_metric_ops
      )
        
    assert mode == tf.estimator.ModeKeys.TRAIN
    
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss,tvars)
    
    global_step = tf.train.get_or_create_global_step()
    
    train_op = params['optimizer'].apply_gradients(zip(grads,tvars),
                                                   global_step=global_step)
    
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)
    
def get_hparams():
  vocab_sizes = {}
  dict_size = [507, 289, 25, 7, 569, 36, 2, 408, 576, 239, 558, 22, 552, 274, 9, 420, 124, 4, 253, 7, 12, 288, 30, 188]
  for i in range(1,14):
    vocab_sizes['I{}'.format(i)] = 11
  vocab_sizes['multi_feats'] = dict_size[0]+5
  for i in range(4,24):
    vocab_sizes['C{}'.format(i)] = dict_size[i-3]+5
    
  field_vocab_mapping = {}
  for name, _ in COLUMNS_MAX_TOKENS:
    field_vocab_mapping[name] = name

  optimizer = tf.train.AdamOptimizer(
      learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')

  return {
    'embedding_dim': 128,
    'vocab_sizes': vocab_sizes,
    # 在这个case中，没有多个field共享同一个vocab的情况，而且field_name和vocab_name相同
    'field_vocab_mapping': field_vocab_mapping,
    'dropout_rate': 0.3,
    'batch_norm': False,
    'hidden_units': [128, 64],
    'optimizer': optimizer
  }
    
if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  
  params = get_hparams()
  
  run_config = tf.estimator.RunConfig(save_checkpoints_secs=120,
                                      keep_checkpoint_max=3)
  
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='models\criteo', params=params,config=run_config)
  
  train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(data_file='dataset/_train.csv',
                                                                n_repeat=10,
                                                                batch_size=1024,
                                                                batches_per_shuffle=10),
                                      max_steps=4096)

  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(data_file='dataset/_test.csv',
                                                                n_repeat=1,
                                                                batch_size=1024,
                                                                batches_per_shuffle=-1),
                                    # steps=50,            # 评估的迭代步数，如果为None，则在整个数据集上评估。
                                    start_delay_secs=5,    #start evaluating after N seconds
                                    throttle_secs=120)     #evaluate every N seconds

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  
# --host=127.0.0.1
# tensorboard logdir是一个目录
# repeat(1)等于不重复