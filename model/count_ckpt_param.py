import sys
import tensorflow as tf
import numpy as np

if len(sys.argv) == 2:
    ckpt_fpath = sys.argv[1]
else:
    print('Usage: python count_ckpt_param.py path-to-ckpt')
    sys.exit(1)

# # Open TensorFlow ckpt
# reader = tf.train.NewCheckpointReader(ckpt_fpath)
#
# print('\nCount the number of parameters in ckpt file(%s)' % ckpt_fpath)
# param_map = reader.get_variable_to_shape_map()
# total_count = 0
# for k, v in param_map.items():
#     if 'Momentum' not in k and 'global_step' not in k:
#         temp = np.prod(v)
#         total_count += temp
#         print('%s: %s => %d' % (k, str(v), temp))
#
# print('Total Param Count: %d' % total_count)

with tf.device('/cpu:0'):
    sess = tf.InteractiveSession()
    graph = tf.get_default_graph()
    saver = tf.train.import_meta_graph(ckpt_fpath +'/mdl.meta')

    checkpoint = '%s/mdl-%d' % (ckpt_fpath, 30000)

    saver.restore(sess, checkpoint)

    # size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
    print sess.run(all_trainable_vars)