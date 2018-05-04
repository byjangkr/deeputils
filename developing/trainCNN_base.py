"""
    Copyright 2018.4. Byeong-Yong Jang
    byjang@cbnu.ac.kr
    This code is for training CNN.


    Input
    -----



    Options
    -------


"""
 

import logging
import os
import pickle
import sys
from optparse import OptionParser

import numpy

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from common import common_io as myio
from common.bnf import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Define function ###
def trainShuff(trainData,trainLabel):
    
    length=trainData.shape[0]
    rng=numpy.random.RandomState(0517)
    train_ind=range(0,length)
    rng.shuffle(train_ind)

    RanTrainData=trainData[train_ind,]
    RanTrainLabel=trainLabel[train_ind,]

    return RanTrainData,RanTrainLabel


def load_spec_vad_data_set(filename,splice_size,spec_stride,class_dict):
    spec_data_set = []
    label_set = []
    with open(filename,'rb') as f:
        (spec_data,vad_data,label) = pickle.load(f)

        # data : splice_size - center(target) frame + splice_size
        # do not paddding
        begi = 0
        endi = begi + splice_size*2 + 1
        while endi < vad_data.shape[0]:
            ispec = spec_data[:,begi:endi]
            centeri = begi + splice_size
            ilabel = (label if vad_data[centeri]==1 else 'sil')

            spec_data_set.append(ispec)
            label_set.append(class_dict[ilabel])

            begi = begi + spec_stride
            endi = begi + splice_size*2 + 1

    spec_data_set = numpy.array(spec_data_set)
    label_set = numpy.array(label_set)
    return spec_data_set, label_set

def mini_batch_from_scp(scplist,mini_batch,splice_size,spec_stride,class_dict,file_pos=0,data_pos=0):
    dataset = []
    labelset = []

    _fpos = file_pos
    _dpos = data_pos

    while (len(dataset) < mini_batch) :
        if not _fpos < len(scplist): # end of iteration
            return numpy.array([]),numpy.array([]),0,0
        nsupple = mini_batch - len(dataset) # number of data to supplement
        (spec_data, label_data) = load_spec_vad_data_set(scplist[_fpos].strip(),splice_size,spec_stride,class_dict)
        ndata = spec_data[_dpos:].shape[0]

        if ndata < nsupple:
            dataset.extend(spec_data[_dpos:])
            labelset.extend(label_data[_dpos:])
            _fpos += 1
            _dpos = 0
        else:
            dpos_ = _dpos + nsupple
            dataset.extend(spec_data[_dpos:dpos_])
            labelset.extend(label_data[_dpos:dpos_])
            _dpos = dpos_

    return numpy.array(dataset), numpy.array(labelset), _fpos, _dpos

def batch_from_scp(scplist,splice_size,spec_stride,class_dict):
    dataset = []
    labelset = []

    _fpos = 0
    _dpos = 0
    while _fpos < len(scplist) :
        (spec_data, label_data) = load_spec_vad_data_set(scplist[_fpos].strip(),splice_size,spec_stride,class_dict)
        dataset.extend(spec_data[_dpos:])
        labelset.extend(label_data[_dpos:])
        _fpos += 1

    return numpy.array(dataset), numpy.array(labelset)

def compute_info(filename, splice_size, spec_stride):
    with open(filename) as f:
        infolist = f.readlines()

    data_cnt = 0
    for info in infolist:
        info_str = "".join(info).strip()
        time_info = int(info_str.split()[3])
        check_info = int(info_str.split()[6])

        if not time_info == check_info: # spec_time_size == vad_size
            print "Warning(compute_info) : wrong info-file -> %s" %(filename)
            return int(-1)

        begi = 0
        endi = begi + splice_size * 2 + 1
        while endi < time_info:
            data_cnt += 1
            begi = begi + spec_stride
            endi = begi + splice_size * 2 + 1

    return int(data_cnt)

### End define function ###

def main():
    usage = "%prog [options] <train-file-scp> <class-dict-file> <directory-for-save-model> <log-file>"
    parser = OptionParser(usage)

    # parser.add_option('--input-dim', dest='inDim',
    #                   help='Input dimension [default: The number of columns in the input data]',
    #                   default=0, type='int')
    # parser.add_option('--num-class', dest='numClass',
    #                   help='The number of classes [default: %default]',
    #                   default=5, type='int')
    parser.add_option('--splice-size', dest='splice_size', help='left-right splice size [default: 5 ]',
                      default=5, type='int')
    parser.add_option('--spec-stride', dest='spec_stride', help='interval between extracted spectrograms [default: 5 ]',
                      default=5, type='int')
    parser.add_option('--num-epoch', dest='num_epoch',
                      help='The number of epoch [default: %default]',
                      default=10, type='int')
    parser.add_option('--minibatch', dest='mini_batch',
                      help='mini-batch size [default: %default]',
                      default=10, type='int')
    parser.add_option('--lr', dest='lr',
                      help='learning rate [default: %default]',
                      default=0.0001, type='float')
    parser.add_option('--keep-prob', dest='keep_prob',
                      help='The probability that each element is kept in dropout [default: %default]',
                      default=0.6, type='float')
    parser.add_option('--val-rate', dest='val_rate',
                      help='validation data rate (%) [default: %default]',
                      default=10, type='int')
    parser.add_option('--val-iter', dest='val_iter',
                      help='Number of iterations to validate the trained model using validation data and recently trained mini-batch data[default: %default]',
                      default=100, type='int')
    parser.add_option('--shuff-epoch', dest='shuff_epoch',
                      help='Number of epochs to shuffle data [default: %default]',
                      default=100, type='int')
    parser.add_option('--save-iter', dest='save_iter',
                      help='Number of iterations to save the training model [default: %default]',
                      default=100, type='int')
    parser.add_option('--mdl-dir', dest='premdl',
                      help='Directory path of pre-model for training',
                      default='', type='string')
    parser.add_option('--info-file', dest='info_file',
                      help='Information file of data',
                      default='', type='string')
    parser.add_option('--active-function', dest='act_func',
                      help='active function relu or sigmoid [default: %default]',
                      default='relu', type='string')

    (o, args) = parser.parse_args()
    (scpfile, classfile, expdir, logfile) = args

    ## set the log 
    mylogger = logging.getLogger("trainDNN")
    mylogger.setLevel(logging.INFO)  # level: debug<info<warning<error<critical, default:warning

    # set print format. In this script: time, message
    formatter = logging.Formatter('Time: %(asctime)s,\nMessage: %(message)s')

    stream_handler = logging.StreamHandler()  # set handler, print to terminal window
    # stream_handler.setFormatter(formatter)
    mylogger.addHandler(stream_handler)

    file_handler = logging.FileHandler(logfile)  # set handler, print to log file
    file_handler.setFormatter(formatter)
    mylogger.addHandler(file_handler)
    ## The end of log setting

    # print command
    # mylogger.info(sys.argv)

    # don't print like format setted in formatter
    file_handler.setFormatter("")
    mylogger.addHandler(file_handler)

    # check the number of input argument
    if len(args) != 4:
        mylogger.info(parser.print_help())
        sys.exit(1)

    save_path = expdir + "/mdl"
    mini_batch = o.mini_batch
    nepoch = o.num_epoch
    lr = o.lr
    val_rate = o.val_rate  # validation data rate (valRate %)
    val_iter = o.val_iter
    save_iter = o.save_iter
    splice_size = o.splice_size
    spec_stride = o.spec_stride # the smaller value, the larger the number of spectrograms extracted from a wavfile
    keep_prob = o.keep_prob
    info_file = o.info_file

    ### End parse options


    ### Read file of train data and label
    # create dict using 'class-dict-file' for converting label(string) to label(int)
    class_dict = {}
    class_info = []
    with open(classfile) as f:
        classlist = f.readlines()

    for line in classlist:
        (labnum, labstr) = line.split()
        class_dict[labstr] = int(labnum)
        class_info.append(int(labnum))
    class_info = numpy.array(class_info)
    nclasses = len(class_info)
    minclass = numpy.min(class_info)

    # read scpfile
    with open(scpfile) as f:
        scplist = f.readlines()

    # current code is approximate, but it's okay because we limited the file size
    mylogger.info("State: read validation data")
    val_inx = int(len(scplist)/100.0*val_rate)
    val_scp = scplist[0:val_inx]

    val_data, val_lab = batch_from_scp(val_scp,splice_size,spec_stride,class_dict)
    val_lab_oh = myio.dense_to_one_hot(val_lab,nclasses,minclass)

    fdim = val_data[0].shape[0]
    tdim = val_data[0].shape[1]

    tr_scp = scplist[val_inx:len(scplist)]

    data_size = -1
    if not info_file == '':
        data_size = compute_info(info_file, splice_size, spec_stride)

    if not data_size == -1:
        tr_data_size = int(data_size - val_data.shape[0])
        niter = int(tr_data_size / mini_batch)
        tr_size_str = "%d"%(tr_data_size)
        niter_str = "%d"%(niter)
    else:
        tr_size_str = "??"
        niter_str = "??"

    ### Main script ###
    mylogger.info('######### Configuration of CNN-model #########')
    mylogger.info('# Dimension of input data = [%d, %d], # of classes = %d' %(fdim,tdim,nclasses))
    # mylogger.info('# Layer size = %d layer, active function: %s',len(hidNode_map), o.actFunc)
    mylogger.info('# Mini-batch size = %d, # of epoch = %d' %(mini_batch,nepoch))
    mylogger.info('# Learning rate = %f, probability of keeping in dropout = %0.1f' %(lr,keep_prob))
    mylogger.info('LOG : train data size = %s, # of iterations = %s'%(tr_size_str,niter_str))
    mylogger.info('LOG : validation data size is %d' % (val_data.shape[0]))

    # with tf.device('/gpu:0'):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.InteractiveSession()

    # make model #
    x = tf.placeholder("float",[None,fdim,tdim], name='x')
    lab_y = tf.placeholder("float", [None, nclasses], name="lab_y")
    keepProb = tf.placeholder("float", name="keepProb")
    bool_dropout = tf.placeholder(tf.bool, name="bool_dropout")
    # bool_batchnorm = tf.placeholder(tf.bool, name="bn_train")

    with tf.name_scope("Reshaping_data") as scope:
        x_img = tf.reshape(x, [-1,fdim,tdim,1])

    with tf.name_scope("Conv1_maxpool_dropout") as scope:
        conv1 = tf.layers.conv2d(inputs=x_img, filters=32, kernel_size=[5, 5],
                                 padding="SAME", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],
                                        padding="SAME", strides=2)
        dropout1 = tf.layers.dropout(inputs=pool1, rate=keepProb, training=bool_dropout)

    with tf.name_scope("Conv2_maxpool_dropout") as scope:
        conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                 padding="SAME", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2],
                                        padding="SAME", strides=2)
        dropout2 = tf.layers.dropout(inputs=pool2, rate=keepProb, training=bool_dropout)

    with tf.name_scope("Conv3_maxpool_dropout") as scope:
        conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                 padding="SAME", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                        padding="SAME", strides=2)
        dropout3 = tf.layers.dropout(inputs=pool3, rate=keepProb, training=bool_dropout)

    with tf.name_scope("Fully_Connected1") as scope:
        flat = tf.reshape(dropout3, [-1, 128*33*3])
        fc1 = tf.layers.dense(inputs=flat, units=2048, activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        dropout4 = tf.layers.dropout(inputs=fc1, rate=keepProb, training=bool_dropout)

    with tf.name_scope("Fully_Connected2") as scope:
        fc2 = tf.layers.dense(inputs=dropout4, units=1048, activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        dropout5 = tf.layers.dropout(inputs=fc2, rate=keepProb, training=bool_dropout)

    with tf.name_scope("Output_layer") as scope:
        out_y = tf.layers.dense(inputs=dropout5, units=nclasses, name="out_y")


    with tf.name_scope("SoftMax") as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_y,labels=lab_y),name="ce")

    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    # begin training

    init = tf.global_variables_initializer()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(out_y,1),tf.argmax(lab_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="acc")


    if o.premdl != "":
        mylogger.info('LOG : train using pre-model -> %s' %(o.premdl) )
        graph = tf.get_default_graph()
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(sess, tf.train.latest_checkpoint(o.premdl))
    else:
        saver = tf.train.Saver(max_to_keep=None)

    saver.save(sess, save_path) # save meta-graph
    mylogger.info("LOG : initial model save with meta-graph -> %s" % save_path)

    epoch=0
    iter=0
    while(epoch<nepoch):
        epoch=epoch+1
        # if epoch%o.shuffEpoch==0:
        #   trData, trLabel=trainShuff(trData, trLabel)
        #   #print 'LOG : data shuffling'

        file_pos = 0
        data_pos = 0
        while True: # do-while loop
            batch_data, batch_lab, file_pos, data_pos = mini_batch_from_scp(tr_scp, mini_batch, splice_size, spec_stride,
                                                                            class_dict, file_pos, data_pos)
            if not batch_data.shape[0] > 0:
                break

            batch_lab_oh = myio.dense_to_one_hot(batch_lab, nclasses, minclass)
            feed_dict = {x: batch_data, lab_y: batch_lab_oh, keepProb: keep_prob, bool_dropout: True}
            sess.run(train_step, feed_dict)
            iter = iter + 1

            # print state of training for validation data and mini-batch
            if (iter % val_iter == 0) | (iter == 1):
                val_acc = []
                val_ce = []
                for i in range(int(val_data.shape[0]/mini_batch+1)):
                    begi = i*mini_batch
                    endi = (i+1)*mini_batch
                    if begi >= val_data.shape[0]:
                        break
                    if endi > val_data.shape[0]:
                        endi = val_data.shape[0]
                    ipred_val = sess.run(out_y, feed_dict={x: val_data[begi:endi],
                                                           keepProb: 1.0, bool_dropout: False})
                    ival_acc = sess.run(accuracy, feed_dict={out_y: ipred_val, lab_y: val_lab_oh[begi:endi]})
                    ival_ce = sess.run(cross_entropy, feed_dict={out_y: ipred_val, lab_y: val_lab_oh[begi:endi]})
                    val_acc.append(ival_acc)
                    val_ce.append(ival_ce)
                val_acc = numpy.mean(numpy.array(val_acc))
                val_ce = numpy.mean(numpy.array(val_ce))

                pred_tr = sess.run(out_y, feed_dict={x: batch_data, keepProb: 1.0, bool_dropout: False})
                tr_acc = sess.run(accuracy, feed_dict={out_y: pred_tr, lab_y: batch_lab_oh})
                tr_ce = sess.run(cross_entropy, feed_dict={out_y: pred_tr, lab_y: batch_lab_oh})

                # set formatter format(time, message)
                file_handler.setFormatter(formatter)
                mylogger.addHandler(file_handler)
                mylogger.info('%d epoch, %d iter (tr/va ce acc) | %f %2.1f%% %f %2.1f%%'
                              % (epoch, iter,tr_ce, (tr_acc*100), val_ce,(val_acc*100)))

                if (iter%save_iter == 0) | (iter == 1): # save parameter
                    saver.save(sess, save_path, global_step=iter, write_meta_graph=False)


    saver.save(sess, save_path, global_step=iter, write_meta_graph=False) # last model save
    mylogger.info("### done \n")

if __name__=="__main__":
    main()



