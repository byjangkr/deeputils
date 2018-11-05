import os, sys
import json
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from deeputils.feature import extract_spec


class Model:
    def __init__(self):
        # Initialize parameter
        self.use_bn = False # use batch normalization
        self.use_dr = True # use dropout
        self.training = tf.placeholder(tf.bool, name='training')
        #self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.keep_prob = 0.6

        # For CNN
        self.img_size = np.array([0, 0], dtype=np.int) # FFT_SiZE x TIME_SPLICE

        # For RNN
        self.rnn_layers = 2
        self.type_rnn = 'GRU' # GRU / LSTM
        self.use_bidirection = True
        self.use_past_out = False


        # For multiscale convolutional layer
        self.use_multiconv = False
        self.fft_size = 512
        self.sample_rate = 16000
        self.multi_kernel_row_size = 64  # number of bins in frequency region
        self.multi_kernel_col_size = 5 # time size of kernel
        self.multi_kernel_n_filters = 3
        self.type_multiscale = 'mel' # mel / chroma

    ''' Set parameters '''
    def read_parameters(self,feat_options_path):
        try:
            with open(feat_options_path,'r') as f:
                FEAT_OPTS = json.load(f)

            self.fft_size = FEAT_OPTS['fft_size']
            self.sample_rate = FEAT_OPTS['sample_rate']
            self.multi_kernel_row_size = FEAT_OPTS['multi_bins']

        except:
            assert('ERROR(CNNModel): wrong feature options file -> %s' %(feat_options_path))

    def set_regularization_parmeters(self,use_bn=False,use_dr=True,keep_prob=0.6):
        self.use_bn = use_bn
        self.use_dr = use_dr
        self.keep_prob = keep_prob
        
    def set_rnn_parameters(self,rnn_layers=2,type_rnn='GRU',use_bidirection=True,use_past_out=False):
        self.rnn_layers = rnn_layers
        self.type_rnn = type_rnn
        self.use_bidirection = use_bidirection
        self.use_past_out = use_past_out

    def set_multiconv_parameters(self,use_multiconv=True,fft_size=512,sample_rate=16000,multi_kernel_row_size=64,
                                 multi_kernel_col_size=5,multi_kernel_n_filters=3,type_multiscale='mel'):
        self.use_multiconv = use_multiconv
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.multi_kernel_row_size = multi_kernel_row_size  # number of bins in frequency region
        self.multi_kernel_col_size = multi_kernel_col_size  # time size of kernel
        self.multi_kernel_n_filters = multi_kernel_n_filters
        self.type_multiscale = type_multiscale  # mel / chroma

    ''' Define functions '''
    def make_convolutional_layer(self,inputs_,filters_, kernel_size_=[3, 3], activation_=tf.nn.relu, padding_='SAME' ,name_='conv'):
        if self.use_bn:
            conv_1 = tf.layers.conv2d(inputs=inputs_, filters=filters_, kernel_size=kernel_size_, padding=padding_, activation=None, name=name_)
            conv_2 = tf.layers.batch_normalization(conv_1,training=self.training)
            conv_out = activation_(conv_2)
        else:
            conv_out = tf.layers.conv2d(inputs=inputs_, filters=filters_, kernel_size=kernel_size_, padding=padding_, activation=tf.nn.relu, name=name_)

        return conv_out

    def make_multiscale_convolutional_layer(self,inputs_):
        part_conv = []
        if self.type_multiscale == 'mel':
            filinx, fildim, multifilt = extract_spec.mel_scale_range(self.fft_size, self.sample_rate, self.multi_kernel_row_size)
        elif self.type_multiscale == 'chroma':
            filinx, fildim, multifilt = extract_spec.chroma_range(self.fft_size, self.sample_rate, self.multi_kernel_row_size)
        else:
            assert('ERROR(CNNModel) : not exist type of scale')

        # padding to maintain time-dimensions
        pad_size = np.int(np.ceil(self.multi_kernel_col_size / 2.0)-1)
        paddings = tf.constant([[0,0],[0,0],[pad_size,pad_size],[0,0]])
        inputs_ = tf.pad(inputs_,paddings,'CONSTANT')

        for ibin in xrange(self.multi_kernel_row_size):
            cname = 'multi_conv_%d' % (ibin)
            init_kernel = tf.constant_initializer(multifilt[ibin, filinx[ibin]])
            part_conv.append(tf.layers.conv2d(inputs=self.slice_freq_range_with_scaled_index(inputs_, filinx, ibin),
                                              filters=self.multi_kernel_n_filters, kernel_size=[fildim[ibin], self.multi_kernel_col_size],
                                              kernel_initializer=init_kernel, padding="valid",
                                              activation=tf.nn.tanh, name=cname))
        multiscale_x = tf.concat(part_conv, 1, name="multiscale_x")
        self.img_size = np.array([self.multi_kernel_row_size,multiscale_x.get_shape()[2].value])
        del part_conv

        out_x = tf.reshape(multiscale_x,[-1,self.multi_kernel_row_size,multiscale_x.get_shape()[2].value,multiscale_x.get_shape()[3].value])
        return out_x

    def slice_freq_range_with_scaled_index(self, inimg_, ininx_, nbin_=0):
        target_boolean = ininx_[nbin_]
        out_img = tf.boolean_mask(inimg_, target_boolean, axis=1)
        return out_img

    def compute_loss(self, logits_, labels_):
        with tf.name_scope("Loss") as scope:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits_, labels=labels_, name="cross_entropy"))

        return loss

    def count_trainable_parameters(self):
        return tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])

    def print_variable_list(self):
        print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def get_kernel_weight(self,target_):
        print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, target_)[0]
        # conv1_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
        # img_kernel = tf.reshape(conv1_kernel,shape=[32,3,3,1])
        # tf.summary.image('kernel image',img_kernel,32)

    ''' Define variable model '''
    def build_cnn_model(self, x, out_dim):
        (_, FFT_SIZE, TIME_SPLICE) = x.get_shape()

        with tf.name_scope("Reshaping_data") as scope:
            x_img = tf.reshape(x, [-1,FFT_SIZE.value,TIME_SPLICE.value,1])
            self.img_size = np.array([FFT_SIZE.value, TIME_SPLICE.value], dtype=np.int)

        if self.use_multiconv:
            with tf.name_scope("Pre_layer_Convolutional_layer_with_multiscale_kernel"):
                x_img = self.make_multiscale_convolutional_layer(x_img)

        with tf.name_scope("Layer_1_Covnolutional_layer_with_maxpooling") as scope:
            conv1 = self.make_convolutional_layer(x_img,32,[3,3], name_='conv1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding='SAME', strides=2)
            if self.use_dr:
                cout1 = tf.layers.dropout(inputs=pool1, rate=(1-self.keep_prob), training=self.training)
            else:
                cout1 = pool1

            self.img_size = np.ceil(self.img_size / 2.0) # for pooling size [2, 2]

        with tf.name_scope("Layer_2_Covnolutional_layer_with_maxpooling") as scope:
            # conv2 = self.make_convolutional_layer(cout1, 64, [3, 3], name_='conv2')
            conv2 = self.make_convolutional_layer(cout1, 16, [3, 3], name_='conv2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding='SAME', strides=2)
            if self.use_dr:
                cout2 = tf.layers.dropout(inputs=pool2, rate=(1-self.keep_prob), training=self.training)
            else:
                cout2 = pool2

            self.img_size = np.ceil(self.img_size / 2.0)  # for pooling size [2, 2]
            
        with tf.name_scope("Layer_3_Covnolutional_layer_with_maxpooling") as scope:
            # conv3 = self.make_convolutional_layer(cout2, 128, [3, 3], name_='conv3')
            conv3 = self.make_convolutional_layer(cout2, 8, [3, 3], name_='conv3')
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding='SAME', strides=2)
            if self.use_dr:
                cout3 = tf.layers.dropout(inputs=pool3, rate=(1-self.keep_prob), training=self.training)
            else:
                cout3 = pool3

            self.img_size = np.ceil(self.img_size / 2.0)  # for pooling size [2, 2]

        with tf.name_scope("Layer_4_Fully_connected_layer") as scope:
            # flat_size = int(self.img_size[0]*self.img_size[1]*128)
            flat_size = int(self.img_size[0] * self.img_size[1] * 8)
            flat4 = tf.reshape(cout3, [-1, flat_size],name='conv_flat')

            if self.use_bn:
                fc4_1 = tf.layers.dense(inputs=flat4, units=2048, activation=None)
                fc4_2 = tf.layers.batch_normalization(fc4_1, training=self.training)
                fc4 = tf.nn.relu(fc4_2)
            else:
                fc4 = tf.layers.dense(inputs=flat4, units=2048, activation=tf.nn.relu)

            if self.use_dr:
                fout4 = tf.layers.dropout(inputs=fc4, rate=(1-self.keep_prob), training=self.training)
            else:
                fout4 = fc4

        with tf.name_scope("Layer_4_Fully_connected_layer") as scope:

            if self.use_bn:
                fc5_1 = tf.layers.dense(inputs=fout4, units=1028, activation=None)
                fc5_2 = tf.layers.batch_normalization(fc5_1, training=self.training)
                fc5 = tf.nn.relu(fc5_2)
            else:
                fc5 = tf.layers.dense(inputs=fout4, units=1028, activation=tf.nn.relu)

            if self.use_dr:
                fout5 = tf.layers.dropout(inputs=fc5, rate=(1-self.keep_prob), training=self.training)
            else:
                fout5 = fc5

        with tf.name_scope("Output_layer") as scope:
            out_y = tf.layers.dense(inputs=fout5, units=out_dim, name="out_y")

        return out_y

    def build_rnn_model(self, x, out_dim):
        (_, FFT_SIZE, TIME_SPLICE) = x.get_shape()

        with tf.name_scope("Reshaping_data") as scope:
            x_img = tf.reshape(x, [-1, FFT_SIZE.value, TIME_SPLICE.value, 1])
            self.img_size = np.array([FFT_SIZE.value, TIME_SPLICE.value], dtype=np.int)

        if self.use_multiconv:
            with tf.name_scope("Pre_layer_Convolutional_layer_with_multiscale_kernel") as scope:
                # self.multi_kernel_n_filters = 1
                x_img = self.make_multiscale_convolutional_layer(x_img)
                x_img = tf.concat(tf.unstack(x_img, axis=3),axis=1)


        with tf.name_scope("Recurrent_layer") as scope:
            x_ref = tf.reshape(x_img,[-1,x_img.get_shape()[1].value,x_img.get_shape()[2].value])
            x_unstack = tf.unstack(x_ref, self.img_size[1], axis=2)

            def _single_cell():
                n_node = 1024
                if self.type_rnn == 'GRU':
                    _cell = rnn.GRUCell(n_node)
                elif self.type_rnn == 'LSTM':
                    _cell = rnn.BasicLSTMCell(n_node, forget_bias=1.0)
                else:
                    assert('ERROR(RNNModel) : not exist type of RNN')

                if self.use_dr:
                    _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=self.keep_prob)

                return _cell

            fw_cell = tf.contrib.rnn.MultiRNNCell([_single_cell() for _ in range(self.rnn_layers)], state_is_tuple=True)

            if self.use_bidirection:
                bw_cell = tf.contrib.rnn.MultiRNNCell([_single_cell() for _ in range(self.rnn_layers)], state_is_tuple=True)
                rnn_outputs, fw_states, bw_states = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x_unstack, dtype=tf.float32)
            else:
                rnn_outputs, current_state = tf.nn.static_rnn(fw_cell, x_unstack, dtype=tf.float32)

            if self.use_past_out:
                rnn_out = tf.concat(rnn_outputs, 1)
            else:
                rnn_out = rnn_outputs[-1]

        with tf.name_scope("Output_layer") as scope:
            out_y = tf.layers.dense(inputs=rnn_out, units=out_dim, name="out_y")

        return out_y


    def build_rnn_attention_model(self, x, out_dim):
        (_, FFT_SIZE, TIME_SPLICE) = x.get_shape()

        with tf.name_scope("Reshaping_data") as scope:
            x_img = tf.reshape(x, [-1, FFT_SIZE.value, TIME_SPLICE.value, 1])
            self.img_size = np.array([FFT_SIZE.value, TIME_SPLICE.value], dtype=np.int)

        if self.use_multiconv:
            with tf.name_scope("Pre_layer_Convolutional_layer_with_multiscale_kernel") as scope:
                # self.multi_kernel_n_filters = 1
                x_img = self.make_multiscale_convolutional_layer(x_img)
                x_img = tf.concat(tf.unstack(x_img, axis=3), axis=1)

        with tf.name_scope("Recurrent_layer") as scope:
            x_ref = tf.reshape(x_img, [-1, x_img.get_shape()[1].value, x_img.get_shape()[2].value])
            x_unstack = tf.unstack(x_ref, self.img_size[1], axis=2)

            def _single_cell():
                n_node = 1024
                if self.type_rnn == 'GRU':
                    _cell = rnn.GRUCell(n_node)
                elif self.type_rnn == 'LSTM':
                    _cell = rnn.BasicLSTMCell(n_node, forget_bias=1.0)
                else:
                    assert ('ERROR(RNNModel) : not exist type of RNN')

                if self.use_dr:
                    _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=self.keep_prob)

                return _cell

            fw_cell = tf.contrib.rnn.MultiRNNCell([_single_cell() for _ in range(self.rnn_layers)],
                                                  state_is_tuple=True)

            if self.use_bidirection:
                bw_cell = tf.contrib.rnn.MultiRNNCell([_single_cell() for _ in range(self.rnn_layers)],
                                                      state_is_tuple=True)
                rnn_outputs, fw_states, bw_states = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x_unstack,
                                                                                 dtype=tf.float32)
            else:
                rnn_outputs, current_state = tf.nn.static_rnn(fw_cell, x_unstack, dtype=tf.float32)

        with tf.name_scope("Attention_layer"):
            # attention part
            # reference by https://github.com/ilivans/tf-rnn-attention/attention.py
            attention_size = 1024
            attention_inputs = tf.transpose(rnn_outputs, [1, 0, 2]) # transpose to [batch_size, time_length, rnn_hidden_node]

            hidden_size = attention_inputs.shape[2].value

            W_ = tf.Variable(tf.random_normal([hidden_size, attention_size],stddev=0.1))
            b_ = tf.Variable(tf.random_normal([attention_size],stddev=0.1))
            u_ = tf.Variable(tf.random_normal([attention_size],stddev=0.1))

            with tf.name_scope('v'):
                # v : [batch_size, time_length, attention_size] = [batch_size, time_length, hidden_node] * [hidden_node, attention_size]
                v = tf.tanh(tf.tensordot(attention_inputs, W_, axes=1) + b_)

            vu = tf.tensordot(v, u_, axes=1, name='vu') # [batch_size, time_length]
            alphas = tf.nn.softmax(vu, name='alphas') # [batch_size, time_length]

            # tf.expand_dims(alphas, -1) = [batch_size, time_length, 1]
            # attention_outputs = [batch_size, hidden_node]
            attention_outputs = tf.reduce_sum(attention_inputs * tf.expand_dims(alphas, -1),1) # sum of time range

            if self.use_dr:
                rnn_out = tf.layers.dropout(inputs=attention_outputs, rate=(1-self.keep_prob), training=self.training)
            else:
                rnn_out = attention_outputs



        with tf.name_scope("Output_layer") as scope:
            out_y = tf.layers.dense(inputs=rnn_out, units=out_dim, name="out_y")

        return out_y

    def build_rnn_attention_feat_model(self, x, out_dim):
        (_, FFT_SIZE, TIME_SPLICE) = x.get_shape()

        with tf.name_scope("Reshaping_data") as scope:
            x_img = tf.reshape(x, [-1, FFT_SIZE.value, TIME_SPLICE.value, 1])
            self.img_size = np.array([FFT_SIZE.value, TIME_SPLICE.value], dtype=np.int)

        with tf.name_scope("Attention_layer"):
            # attention part
            # reference by https://github.com/ilivans/tf-rnn-attention/attention.py
            attention_size = 1024
            # attention_inputs = tf.transpose(x, [1, 0, 2]) # transpose to [batch_size, features, time_length]
            attention_inputs = x

            hidden_size = attention_inputs.shape[2].value

            W_ = tf.Variable(tf.random_normal([hidden_size, attention_size],stddev=0.1))
            b_ = tf.Variable(tf.random_normal([attention_size],stddev=0.1))
            u_ = tf.Variable(tf.random_normal([attention_size],stddev=0.1))

            with tf.name_scope('v'):
                # v : [batch_size, time_length, attention_size] = [batch_size, time_length, hidden_node] * [hidden_node, attention_size]
                v = tf.tanh(tf.tensordot(attention_inputs, W_, axes=1) + b_)

            vu = tf.tensordot(v, u_, axes=1, name='vu') # [batch_size, time_length]
            alphas = tf.nn.softmax(vu, name='alphas') # [batch_size, time_length]

            # tf.expand_dims(alphas, -1) = [batch_size, time_length, 1]
            # attention_outputs = [batch_size, hidden_node]
            # attention_outputs = tf.reduce_sum(attention_inputs * tf.expand_dims(alphas, -1),1) # sum of time range
            attention_outputs = attention_inputs * tf.expand_dims(alphas, -1) # [batch_size, time_length, features]


            rnn_input= attention_outputs
            print rnn_input

        with tf.name_scope("Recurrent_layer") as scope:
            # x_ref = tf.reshape(x_img, [-1, x_img.get_shape()[1].value, x_img.get_shape()[2].value])
            x_unstack = tf.unstack(rnn_input, self.img_size[1], axis=2)
            # print x_ref, rnn_input


            def _single_cell():
                n_node = 1024
                if self.type_rnn == 'GRU':
                    _cell = rnn.GRUCell(n_node)
                elif self.type_rnn == 'LSTM':
                    _cell = rnn.BasicLSTMCell(n_node, forget_bias=1.0)
                else:
                    assert ('ERROR(RNNModel) : not exist type of RNN')

                if self.use_dr:
                    _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=self.keep_prob)

                return _cell

            fw_cell = tf.contrib.rnn.MultiRNNCell([_single_cell() for _ in range(self.rnn_layers)],
                                                  state_is_tuple=True)

            if self.use_bidirection:
                bw_cell = tf.contrib.rnn.MultiRNNCell([_single_cell() for _ in range(self.rnn_layers)],
                                                      state_is_tuple=True)
                rnn_outputs, fw_states, bw_states = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x_unstack,
                                                                                 dtype=tf.float32)
            else:
                rnn_outputs, current_state = tf.nn.static_rnn(fw_cell, x_unstack, dtype=tf.float32)

            if self.use_past_out:
                rnn_out = tf.concat(rnn_outputs, 1)
            else:
                rnn_out = rnn_outputs[-1]

        with tf.name_scope("Output_layer") as scope:
            out_y = tf.layers.dense(inputs=rnn_out, units=out_dim, name="out_y")

        return out_y

    def build_longtime_melcnn_rnn_model(self, x, out_dim, rnn_seq_size):
        (_, FFT_SIZE, TOTAL_TIME_SPLICE) = x.get_shape()

        TIME_SPLICE = TOTAL_TIME_SPLICE / rnn_seq_size

        with tf.name_scope("Reshaping_data") as scope:
            x_unpack_time = tf.reshape(tf.transpose(x,[0,2,1]),[-1,TIME_SPLICE,FFT_SIZE]) # [batch_size*rnn_seq_size, time_splice, fft_size]
            # x2 = tf.reshape(x_unpack_time,[-1,TOTAL_TIME_SPLICE,FFT_SIZE])
            x_img = tf.reshape(tf.transpose(x_unpack_time,[0,2,1]), [-1, FFT_SIZE.value, TIME_SPLICE.value, 1])
            self.img_size = np.array([FFT_SIZE.value, TIME_SPLICE.value], dtype=np.int)

        with tf.name_scope("Pre_layer_Convolutional_layer_with_multiscale_kernel") as scope:
            # self.multi_kernel_n_filters = 1
            x_img = self.make_multiscale_convolutional_layer(x_img)
            # x_img = tf.concat(tf.unstack(x_img, axis=3),axis=1)
            self.img_size = np.array([x_img.get_shape()[1].value, x_img.get_shape()[2].value], dtype=np.int)
            # x_img = tf.reshape(x_img,[-1, self.img_size[0], self.img_size[1], 1])

        with tf.name_scope("Embedding_Covnolutional_layer_with_maxpooling1") as scope:
            conv1 = self.make_convolutional_layer(x_img, 32, [3, 3], name_='conv1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding='SAME', strides=2)
            if self.use_dr:
                cout1 = tf.layers.dropout(inputs=pool1, rate=(1 - self.keep_prob), training=self.training)
            else:
                cout1 = pool1

            # cout1_img = tf.concat(tf.unstack(cout1, axis=3), axis=1)

            self.img_size = np.array([cout1.get_shape()[1].value, cout1.get_shape()[2].value], dtype=np.int)


        with tf.name_scope("Embedding_Covnolutional_layer_with_maxpooling2") as scope:
            conv2 = self.make_convolutional_layer(cout1, 16, [3, 3], name_='conv2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding='SAME', strides=2)
            if self.use_dr:
                cout2 = tf.layers.dropout(inputs=pool2, rate=(1 - self.keep_prob), training=self.training)
            else:
                cout2 = pool2

            # cout2_img = tf.concat(tf.unstack(cout2, axis=3), axis=1)

            self.img_size = np.array([cout2.get_shape()[1].value, cout2.get_shape()[2].value], dtype=np.int)


        with tf.name_scope("Embedding_Covnolutional_layer_with_maxpooling3") as scope:
            conv3 = self.make_convolutional_layer(cout2, 8, [3, 3], name_='conv3')
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding='SAME', strides=2)
            if self.use_dr:
                cout3 = tf.layers.dropout(inputs=pool3, rate=(1 - self.keep_prob), training=self.training)
            else:
                cout3 = pool3

            cout3_img = tf.concat(tf.unstack(cout2, axis=3), axis=1)

            self.img_size = np.array([cout3_img.get_shape()[1].value, cout3_img.get_shape()[2].value], dtype=np.int)


        with tf.name_scope("Recurrent_layer") as scope:
            x_ref = tf.reshape(cout3_img,[-1,rnn_seq_size,self.img_size[0],self.img_size[1]]) # [batch_size, rnn_seq_size, n_bin*n_filter, time_splice]
            x_pack_time = tf.transpose(tf.reshape(x_ref, [-1,rnn_seq_size,self.img_size[0]*self.img_size[1]]),[0,2,1]) # [batch_size, embedding_melfeat, rnn_seq_size]
            x_unstack = tf.unstack(x_pack_time, rnn_seq_size, axis=2)

            def _single_cell():
                n_node = 1024
                if self.type_rnn == 'GRU':
                    _cell = rnn.GRUCell(n_node)
                elif self.type_rnn == 'LSTM':
                    _cell = rnn.BasicLSTMCell(n_node, forget_bias=1.0)
                else:
                    assert('ERROR(RNNModel) : not exist type of RNN')

                if self.use_dr:
                    _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=self.keep_prob)

                return _cell

            fw_cell = tf.contrib.rnn.MultiRNNCell([_single_cell() for _ in range(self.rnn_layers)], state_is_tuple=True)

            if self.use_bidirection:
                bw_cell = tf.contrib.rnn.MultiRNNCell([_single_cell() for _ in range(self.rnn_layers)], state_is_tuple=True)
                rnn_outputs, fw_states, bw_states = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x_unstack, dtype=tf.float32)
            else:
                rnn_outputs, current_state = tf.nn.static_rnn(fw_cell, x_unstack, dtype=tf.float32)

            if self.use_past_out:
                rnn_out = tf.concat(rnn_outputs, 1)
            else:
                rnn_out = rnn_outputs[-1]

        with tf.name_scope("Output_layer") as scope:
            out_y = tf.layers.dense(inputs=rnn_out, units=out_dim, name="out_y")

        return out_y


if __name__ == '__main__':

    ''' Example of build model'''

    x = tf.placeholder(tf.float32,shape=[None,257,101])
    mdl = Model()
    mdl.set_regularization_parmeters(use_bn=False,use_dr=True)
    mdl.set_multiconv_parameters(fft_size=512,sample_rate=16000,multi_kernel_row_size=64,multi_kernel_col_size=5,multi_kernel_n_filters=3,type_multiscale='mel')
    # mdl.set_rnn_parameters(rnn_layers=2,type_rnn='GRU',use_bidirection=True,use_past_out=False)
    # out_y = mdl.build_longtime_melcnn_rnn_model(x,out_dim=2,rnn_seq_size=25)
    out_y = mdl.build_cnn_model(x,out_dim=2)


    # data = np.append(np.ones([1,257,525]),np.ones([1,257,525])*2,axis=0)

    # print data
    # print out_y.name, x.name
    # out_y_softmax = tf.nn.softmax(out_y)
    # loss = mdl.compute_loss(out_y,out_y)

    sess = tf.Session()
    # print sess.run(out_y,feed_dict={x:data})
    print sess.run(mdl.count_trainable_parameters())
    # #
    # # print sess.run([mdl.training], feed_dict={mdl.training:True})


    # mdl.print_variable_list()




