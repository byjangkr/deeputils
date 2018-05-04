#!/usr/bin/env python

import tensorflow.python.platform
import tensorflow as tf
import numpy 
import sys
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

###
### Parse options
###
from optparse import OptionParser
usage="%prog [options] <train-data-file> <train-label-file> <test-data-file> <test-label-file> <test-predict-label-file>"
parser = OptionParser(usage)

parser.add_option('--feat-dim', dest='featDim', 
                   help='Feature dimension [default: %default]', 
                   default=20, type='int');

parser.add_option('--num-class', dest='numClass', 
                   help='The number of classes [default: %default]', 
                   default=5, type='int');

(o,args) = parser.parse_args()
if len(args) != 5 : 
  parser.print_help()
  sys.exit(1)
  
#(trDataFile, trLabelFile, tsDataFile, tsLabelFile) = map(int,args);
(trDataFile, trLabelFile, tsDataFile, tsLabelFile, tsPredFile) = args


### End parse options 

### Define function ###

def dense_to_one_hot(labels_dense, num_classes=5):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def next_batch(pre_index, batch_size, data_size):
  """Return the next `batch_size` examples from this data set."""
  start = pre_index
  check_index = start + batch_size
  if  check_index > data_size:
    # Start next epoch
    start = 0

  end = start + batch_size
  return start, end
### End define function ###


### Read file of train/test data and label

d1 = open(trDataFile)
buf = d1.read()
trData = numpy.fromstring(buf, dtype=numpy.float32, sep=' ')
trData = trData.reshape(len(trData)/o.featDim,o.featDim)
d1.close()

l1 = open(trLabelFile)
buf = l1.read()
trLabel = numpy.fromstring(buf, dtype=numpy.uint8, sep=' ')
trLabel = dense_to_one_hot(trLabel,o.numClass)
l1.close()

d2 = open(tsDataFile)
buf = d2.read()
tsData = numpy.fromstring(buf, dtype=numpy.float32, sep=' ')
tsData = tsData.reshape(len(tsData)/o.featDim,o.featDim)
d2.close()

l2 = open(tsLabelFile)
buf = l2.read()
tsLabel = numpy.fromstring(buf, dtype=numpy.uint8, sep=' ')
tsLabel = dense_to_one_hot(tsLabel,o.numClass)
l2.close()

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = o.featDim
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = len(trData)/o.featDim


class RNNModel(object):
  """The RNN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, o.numClass])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.split(
          1, num_steps, tf.nn.embedding_lookup(embedding, self._input_data))
      inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    if is_training and config.keep_prob < 1:
      inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    # outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    states = []
    state = self._initial_state

    with tf.variable_scope("RNN"):
      for time_step, input_ in enumerate(inputs):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        
        (cell_output, state) = cell(input_, state)
        outputs.append(cell_output)
        states.append(state)
        

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    
    logits = tf.nn.xw_plus_b(output,
                             tf.get_variable("softmax_w", [size, vocab_size]),
                             tf.get_variable("softmax_b", [vocab_size]))
    loss = seq2seq.sequence_loss_by_example([logits],
                                            [tf.reshape(self._targets, [-1])],
                                            [tf.ones([batch_size * num_steps])],
                                            vocab_size)
    self._cost = cost = tf.div(tf.reduce_sum(loss), batch_size)
    self._final_state = states[-1]

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps
    
    #print("%s" %(x[1]))
    #print("%s" %(y[1]))

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


### Main script ###
sess = tf.Session()
m = RNNModel(is_training=True, config=SmallConfig())
init = tf.initialize_all_variables()
sess.run(init)

epoch_size = ((len(trData) // m.batch_size) - 1) // m.num_steps
state = m.initial_state
pre_index = 0
for i in range(1):
  beg_index, end_index = next_batch(pre_index, m.batch_size, len(trData))
  pre_index = end_index
  feed_dict = {m.input_data: trData[beg_index:end_index], m.targets: trLabel[beg_index:end_index], m.initial_state: state}
  #print(beg_index,end_index)
  cost, state, _ = sess.run([m.cost, m.final_state, m.train_op], feed_dict)


