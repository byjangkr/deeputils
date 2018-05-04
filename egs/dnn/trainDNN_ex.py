#!/usr/bin/env python

# Title : trainDNN_ex.py
# Autor : Byeong-Yong Jang
# Data : 2018.04.02
# E-mail : darkbulls44@gmail.com
# Github : https://github.com/byjangkr
# Copyright (c) 2018. All rights reserved.

import tensorflow as tf
import numpy 
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from common import common_io as myio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###
### Parse options
###
from optparse import OptionParser
usage="%prog [options] <train-data-file> <train-label-file> <directory-for-save-model>"
parser = OptionParser(usage)

parser.add_option('--input-dim', dest='inDim', 
                   help='Input dimension [default: The number of columns in the input data]', 
                   default=0, type='int');

parser.add_option('--num-class', dest='numClass', 
                   help='The number of classes [default: %default]', 
                   default=5, type='int');

parser.add_option('--num-epoch', dest='numEpoch', 
                   help='The number of epoch [default: %default]', 
                   default=10, type='int');

parser.add_option('--minibatch', dest='mBatch', 
                   help='mini-batch size [default: %default]', 
                   default=10, type='int');

parser.add_option('--lr', dest='lr', 
                   help='learning rate [default: %default]', 
                   default=0.0001, type='float');

parser.add_option('--keep-prob', dest='kProb', 
                   help='The probability that each element is kept in dropout [default: %default]', 
                   default=0.6, type='float');

parser.add_option('--valRate', dest='valRate', 
                   help='validation data rate (%) [default: %default]', 
                   default=10, type='int');

parser.add_option('--valEpoch', dest='valEpoch', 
                   help='Number of epochs to validate the trained model [default: %default]', 
                   default=100, type='int');

parser.add_option('--shuff-epoch', dest='shuffEpoch', 
                   help='Number of epochs to shuffle data [default: %default]', 
                   default=100, type='int');

parser.add_option('--save-epoch', dest='saveEpoch', 
                   help='Number of epochs to save the training model [default: %default]', 
                   default=100, type='int');

parser.add_option('--mdl-dir', dest='premdl', 
                   help='Directory path of pre-model for training', 
                   default='', type='string');


(o,args) = parser.parse_args()
if len(args) != 3 : 
  parser.print_help()
  sys.exit(1)
  
#(trDataFile, trLabelFile, tsDataFile, tsLabelFile) = map(int,args);
(trDataFile, trLabelFile, expdir) = args

save_path = expdir + "/dnnmdl"
miniBatch = o.mBatch
nEpoch = o.numEpoch
lr = o.lr
valRate = o.valRate # validation data rate (valRate %) 


hidNode_map = {
  'hid1':100,
  'hid2':100,
  'hid3':100,
  'hid4':50,
  'hid5':10
}
hidNode1 = hidNode_map['hid1']
hidNode2 = hidNode_map['hid2']
hidNode3 = hidNode_map['hid3']
hidNode4 = hidNode_map['hid4']
hidNode5 = hidNode_map['hid5']

### End parse options 


### Define function ###
def weight_variable(shape,name=""):
  initializer = tf.contrib.layers.xavier_initializer()
  if name == "":
    return tf.Variable(initializer(shape))
  else:
    return tf.Variable(initializer(shape),name=name)


def bias_variable(shape,name=""):
  initial = tf.random_normal(shape)
  if name == "":
    return tf.Variable(initial)
  else:
    return tf.Variable(initial, name=name)

def build_layer(pre_node,cur_node,pre_layer,act_func='relu'):
  W = weight_variable([pre_node, cur_node])
  b = bias_variable([cur_node])
  if act_func == 'sigmoid':
    h = tf.sigmoid(tf.matmul(pre_layer,W) + b)
  elif act_func == 'relu':
    h = tf.nn.relu(tf.matmul(pre_layer,W) + b)
  else:
    h = tf.sigmoid(tf.matmul(pre_layer,W) + b)

  return W,b,h

def trainShuff(trainData,trainLabel):
    
    length=trainData.shape[0]
    rng=numpy.random.RandomState(0517)
    train_ind=range(0,length)
    rng.shuffle(train_ind)

    RanTrainData=trainData[train_ind,]
    RanTrainLabel=trainLabel[train_ind,]

    return RanTrainData,RanTrainLabel

def next_batch(pre_index, batch_size, data_size):
  """Return the next `batch_size` examples from this data set."""
# Usage
#pre_index = 0
#for i in range(100000):
#  beg_index, end_index = next_batch(pre_index, miniBatch, len(trData))
#  pre_index = end_index
#  feed_dict = {x: trData[beg_index:end_index], y_: trLabel[beg_index:end_index]}
  #print(beg_index,end_index)
#  sess.run(train_step, feed_dict)

  start = pre_index
  check_index = start + batch_size
  if  check_index > data_size:
    # Start next epoch
    start = 0

  end = start + batch_size
  return start, end
### End define function ###


### Read file of train data and label

oriTrData, o.inDim = myio.read_data_file(trDataFile,o.inDim)
oriTrLabel_tmp = myio.read_label_file(trLabelFile)
oriTrLabel = myio.dense_to_one_hot(oriTrLabel_tmp,o.numClass)

oriTrData, oriTrLabel=trainShuff(oriTrData, oriTrLabel) # shuffling

valInx = oriTrData.shape[0]/100*valRate
valData = oriTrData[0:valInx]
valLabel = oriTrLabel[0:valInx]

trData = oriTrData[valInx+1:oriTrData.shape[0]]
trLabel = oriTrLabel[valInx+1:oriTrLabel.shape[0]]

totalBatch = trData.shape[0]/miniBatch


### Main script ###
print '######### Configuration of DNN-model #########'
print '# Dimension of input data = %d, # of classes = %d' %(o.inDim,o.numClass)
print '# Mini-batch size = %d, # of epoch = %d' %(miniBatch,nEpoch)
print '# Learning rate = %f, probability of keeping in dropout = %0.1f' %(lr,o.kProb)

print 'LOG : train data size = %d, # of iterations = %d' %(trData.shape[0],totalBatch)
print 'LOG : validation data size is %d (%d%%) of %d training data' %(valInx,valRate,(oriTrData.shape[0]+1))


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# make model #
x = tf.placeholder("float", [None, o.inDim], name="x")
lab_y = tf.placeholder("float", [None, o.numClass], name="lab_y")
keepProb = tf.placeholder("float", name="keepProb")

[W1, b1, h1] = build_layer(o.inDim, hidNode1, x, 'relu') 
h1Drop = tf.nn.dropout(h1, keepProb)

[W2, b2, h2] = build_layer(hidNode1, hidNode2, h1Drop, 'relu') 
h2Drop = tf.nn.dropout(h2, keepProb)

[W3, b3, h3] = build_layer(hidNode2, hidNode3, h2Drop, 'relu') 
h3Drop = tf.nn.dropout(h3, keepProb)

[W4, b4, h4] = build_layer(hidNode3, hidNode4, h3Drop, 'relu') 
h4Drop = tf.nn.dropout(h4, keepProb)

[W5, b5, h5] = build_layer(hidNode4, hidNode5, h4Drop, 'relu')
h5Drop = tf.nn.dropout(h5, keepProb)

W_last = weight_variable([hidNode5, o.numClass],"W_last")
b_last = bias_variable([o.numClass],"b_last")
out_y = tf.add(tf.matmul(h5Drop,W_last,name="mm_last"),b_last,name="out_y")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_y,labels=lab_y),name="ce")
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# begin training

init = tf.global_variables_initializer()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(out_y,1),tf.argmax(lab_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="acc")


if o.premdl != "":
  print 'LOG : train using pre-model -> %s' %(o.premdl) 
  graph = tf.get_default_graph()
  saver = tf.train.Saver(max_to_keep=None)
  saver.restore(sess, tf.train.latest_checkpoint(o.premdl))
else:
  saver = tf.train.Saver(max_to_keep=None)

saver.save(sess, save_path) # save meta-graph
print("LOG : initial model save with meta-graph -> %s" % save_path)

epoch=0
while(epoch<nEpoch):
  epoch=epoch+1
  if epoch%o.shuffEpoch==0:	
    trData, trLabel=trainShuff(trData, trLabel)
    #print 'LOG : data shuffling'

  for train_index in xrange(totalBatch):
    feed_dict={
      x: trData[train_index*miniBatch: (train_index+1)*miniBatch],
      lab_y: trLabel[train_index*miniBatch: (train_index+1)*miniBatch],
      keepProb: o.kProb
      }

    sess.run(train_step,feed_dict)

  if epoch%o.valEpoch==0: # print state of training for validation data
    pred_val = sess.run(out_y,feed_dict={x: valData, lab_y:valLabel, keepProb:1.0})
    val_acc = sess.run(accuracy, feed_dict={out_y: pred_val, lab_y: valLabel})
    val_ce = sess.run(cross_entropy, feed_dict={out_y: pred_val, lab_y: valLabel})
    print 'Training %d epoch : ce = %f, acc = %2.1f%% ' %(epoch,(val_ce/valData.shape[0]),(val_acc*100))

  if epoch%o.saveEpoch==0: # save parameter
    saver.save(sess, save_path, global_step=epoch,write_meta_graph=False)

saver.save(sess, save_path, global_step=epoch,write_meta_graph=False) # last model save
print "### done \n"




