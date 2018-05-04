#!/usr/bin/env python

import numpy 

### Define function ###

def read_data_file(filename, dim=0):
  # read data file
  # 0.1 0.4 0.2 ... 0.5
  #       .
  #       .
  #       .
  # 0.3 0.5 1.3 ... 3.4
  # if dim is 0, then dim is number of column in data file 

  f = open(filename)

  if dim == 0:
    bufline = f.readline()
    dim1 = len(numpy.fromstring(bufline, dtype=numpy.float32, sep=' '))
    bufline = f.readline()
    dim2 = len(numpy.fromstring(bufline, dtype=numpy.float32, sep=' '))
    if dim1 != dim2:
      assert("ERROR(common_io:read_data_file): do not eqaul the number of column between row1 and row2")

    dim = dim1

  #print 'LOG : read file -> %s, dim=%d' % (filename, dim)
  f.seek(0)
  buf = f.read()
  data = numpy.fromstring(buf, dtype=numpy.float32, sep=' ')
  data = data.reshape(len(data)/dim,dim)
  return data, dim

  # end of read_data_file()

def read_label_file(filename):
  # read label file
  # 0
  # 1
  # 2
  # .
  # .
  # .
  # 9

  f = open(filename)
  #print 'LOG : read file -> %s' % (filename)
  buf = f.read()
  lab = numpy.fromstring(buf, dtype=numpy.uint8, sep=' ')
  return lab

  # end of read_label_file()
  
def dense_to_one_hot(labels_dense, num_classes=0):
  """Convert class labels from scalars to one-hot vectors."""
  min_val = numpy.ndarray.min(labels_dense)
  max_val = numpy.ndarray.max(labels_dense)
  pred_num_classes = max_val - min_val + 1
  if num_classes == 0:
    num_classes = pred_num_classes

  if pred_num_classes != num_classes :
    print "WARNING : # of classess is not match between %d(input) and %d(predicted)" %(num_classes,pred_num_classes)

  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel() - min_val] = 1
  return labels_one_hot
  # end of dense_to_one_hot()

def write_predicted_prob(data,filename):
  """Write predicted prob data of dnn"""
  f = open(filename,'w')

  for j in range(data.shape[0]):
    for k in range(data.shape[1]):
      buf = "%f " %(data[j][k]) 
      f.write(buf)
    f.write('\n')
  f.close()

def write_predicted_lab(data,filename):
  """Write predicted lab data of dnn"""
  f = open(filename,'w')

  for j in range(data.shape[0]):
    buf = "%d " %(data[j]) 
    f.write(buf)

  f.close()


### End of define function ###


