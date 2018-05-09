#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf

#dirctory path to the tfrecords
tf.app.flags.DEFINE_string('data_dir_mo',
    '../../../data/tfrecord/',
    'Path to the TFRecord data directory.')

#number of images to used for single learning
tf.app.flags.DEFINE_integer('batch_size', 20,
    'Batch size Must divide evenly into the dataset sizes.')

#bias default is 16
tf.app.flags.DEFINE_integer('bias', 16, 'bias default is 16')

#conv_loops=4
tf.app.flags.DEFINE_integer('conv_loops', 4, 'conv_loops=4')

#image_size
tf.app.flags.DEFINE_integer('image_size', 96, 'image size = 28')

#channels default will be color imgae = 3
tf.app.flags.DEFINE_integer('channels', 3, 'default will be color imgae = 3')

#for testing
tf.app.flags.DEFINE_integer('num_classes', 4, 'number of classes/labels')

#dirctory where to store checkpoint files
tf.app.flags.DEFINE_string('checkpoint_dir', '../../../log/checkpoint/',
    'Path to read model checkpoint.')

#number of files to keep the learned data
tf.app.flags.DEFINE_integer('max_to_keep', 10, 'Number of files to keep the learned data')

#number of learning steps
tf.app.flags.DEFINE_integer('max_steps', 10, 'Number of steps to run trainer.')

#where to store timeline logs
tf.app.flags.DEFINE_string('log_dir', '../../../log/log/',
    'Path to read model log.')

#To restudy the old model or not 0: false, 1: true
tf.app.flags.DEFINE_integer('restudy_old_model', 1, 'To restudy the old model or not')

#create protocal buffer file from the studied graph or not 0: false, 1: true
tf.app.flags.DEFINE_integer('save_as_pb', 1,
    'create protocal buffer file from the studied graph or not')

#testing image dirctory
tf.app.flags.DEFINE_string('test_image_dir', '../../../data/image/A/',
    'testing image dirctory')

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
