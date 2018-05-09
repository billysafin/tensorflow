#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf

#image_size
tf.app.flags.DEFINE_integer('image_size', 96, 'image size = 28')

#channels default will be color imgae = 3
tf.app.flags.DEFINE_integer('channels', 3, 'default will be color imgae = 3')

#bias default is 16
tf.app.flags.DEFINE_integer('bias', 16, 'bias default is 16')

#conv_loops=4
tf.app.flags.DEFINE_integer('conv_loops', 4, 'conv_loops=4')

#dirctory where to store checkpoint files
tf.app.flags.DEFINE_string('checkpoint_dir', '../../../log/checkpoints/',
    'Path to read model checkpoint.')

#where to store timeline logs
tf.app.flags.DEFINE_string('log_dir', '../../../log/log/',
    'Path to read model log.')

#4
tf.app.flags.DEFINE_integer('num_classes', 4, 'number of classes/labels')

#Final bias
tf.app.flags.DEFINE_integer('final_bias', 1024, 'final bias')

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = FLAGS.image_size * FLAGS.image_size * FLAGS.channels

LABELS = {
    0   : 'A'
    , 1 : 'B'
    , 2 : 'C'
    , 3 : 'D'
}
