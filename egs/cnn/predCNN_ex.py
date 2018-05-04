#!/usr/bin/env python

# Title : predCNN_ex.py
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
from sample_data.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.InteractiveSession()

###
### Parse options
###
from optparse import OptionParser
usage="%prog [options] <test-data-file> [<test-label-file>] <directory of model>"
parser = OptionParser(usage)

parser.add_option('--out-predprob', dest='predprob', 
                   help='Output file of predicted probability', 
                   default='', type='string');

parser.add_option('--out-predlab', dest='predlab', 
                   help='Output file of predicted label', 
                   default='', type='string');

parser.add_option('--min-predlab', dest='minpredlab', 
                   help='Minimum value of predicted label [default: %default]', 
                   default=0, type='int');

parser.add_option('--checkpoint', dest='checkpoint', 
                   help='Checkpoint number to use DNN model [default: last checkpoint]', 
                   default=0, type='int');

print '######### Predict data with DNN-model #########'

(o,args) = parser.parse_args()
if len(args) < 2 : 
  parser.print_help()
  sys.exit(1)
elif len(args) == 3 :
  (data_file, label_file, mdldir) = args
  in_lab = True
else :
  (data_file, mdldir) = args
  in_lab = False


### Start main ### 

with tf.device('/cpu:0'):
# load meta graph and restore weights
  print 'LOG : load model -> %s' %(mdldir+'/cnnmdl.meta')
  graph = tf.get_default_graph()
  saver = tf.train.import_meta_graph(mdldir+'/cnnmdl.meta')
  if o.checkpoint > 0 :
    checkpoint = '%s/dnnmdl-%d' %(mdldir,o.checkpoint)
  else :
    checkpoint = tf.train.latest_checkpoint(mdldir)

  saver.restore(sess, checkpoint)
  print 'LOG : checkpoint -> %s ' %(checkpoint)

  x = graph.get_tensor_by_name("x:0")
  lab_y = graph.get_tensor_by_name("lab_y:0")
  out_y = graph.get_tensor_by_name("out_y:0")
  keepProb = graph.get_tensor_by_name("keepProb:0")
  ce = graph.get_tensor_by_name("ce:0")
  acc = graph.get_tensor_by_name("acc:0")


  mnist = input_data.read_data_sets(data_file, one_hot=True)
  num_data = mnist.test.num_examples
  data = mnist.test.next_batch(num_data)
  test_data = data[0]
  featdim = test_data.shape[1]
  test_lab_ot = data[1]
  o.numClass = test_lab_ot.shape[1]
  in_lab = True
  #test_data, featdim = myio.read_data_file(data_file)
  print 'LOG : predict probability using CNN-model'

  pred_data =  sess.run(out_y,feed_dict ={x:test_data, keepProb:1.0})

if in_lab:
#  test_lab = myio.read_label_file(label_file)
#  test_lab_ot = myio.dense_to_one_hot(test_lab,2)

  pred_acc = sess.run(acc, feed_dict={out_y: pred_data, lab_y: test_lab_ot})
  pred_ce = sess.run(ce, feed_dict={out_y: pred_data, lab_y: test_lab_ot})
  print 'Results : '
  print '  # of data = %d' %(pred_data.shape[0])
  print '  average of cross entropy = %f' %(pred_ce)
  print '  accuracy = %2.1f%%' %(pred_acc*100)
  print '### done\n'

if o.predprob != '':
  print 'LOG : write predicted probability -> %s' %(o.predprob)
  myio.write_predicted_prob(pred_data,o.predprob)

if o.predlab != '':
  print 'LOG : write predicted label -> %s' %(o.predlab)
  pred_lab = numpy.argmax(pred_data, axis=1) + o.minpredlab
  myio.write_predicted_lab(pred_lab,o.predlab)

### End main ###
