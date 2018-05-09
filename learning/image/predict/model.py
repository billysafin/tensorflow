#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from . import setting

def inference(image_batch_flatten, keep_prob=0.5, reuse=False):
    '''
    obtain the inference
    Args:
        image_batch_flatten = [batch_size, (height * width * channels)] tensorflow object
        keep_prob = drop_out 0.5
        reuse = reuse the variable again
    Return:
        inferernce [batch_size, ]
    '''
    with tf.variable_scope('inference'):
        convs = []
        pools = []
        i = 1

        def _variable_with_weight_decay(name, shape, stddev, wd):
            var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
            if wd:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            return var

        while i <= setting.FLAGS.conv_loops:
            if i == 1:
                channels = setting.FLAGS.channels
                biases_shape = setting.FLAGS.bias
                conv_target = tf.reshape(image_batch_flatten, 
                    [-1, setting.FLAGS.image_size, setting.FLAGS.image_size, setting.FLAGS.channels]
                )
            else:
                channels = biases_shape
                biases_shape = channels * 2
                conv_target = pools[-1]

            conv_i = 'conv_' + str(i)
            with tf.variable_scope(conv_i, reuse=reuse) as scope:
                weight = tf.get_variable('weights', 
                    [5, 5, channels, biases_shape], 
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv = tf.nn.conv2d(
                    conv_target, 
                    weight, 
                    [1, 1, 1, 1], 
                    padding='SAME')
                biases = tf.get_variable('biases', 
                    shape=[biases_shape], 
                    initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(conv, biases)
                convs.append(tf.nn.dropout(tf.nn.relu(bias, name=scope.name), keep_prob))
            pools.append(
                tf.nn.max_pool(
                    convs[-1], 
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME'))
            i += 1

        with tf.variable_scope('fc_1', reuse=reuse) as scope:
            dim = 1
            for d in pools[-1].get_shape()[1:].as_list():
                dim *= d
            
            c_weight_1 = _variable_with_weight_decay(
                'c_weight_1', 
                shape=[dim, setting.FLAGS.final_bias], 
                stddev=0.01, 
                wd=0.005)
            c_bias_1 = tf.get_variable(
                'c_bias_1', 
                shape=[setting.FLAGS.final_bias], 
                initializer=tf.constant_initializer(0.0))  
            pool_flatten = tf.reshape(pools[-1], [-1, dim])
            h_fc1 = tf.nn.relu(tf.matmul(pool_flatten, c_weight_1) + c_bias_1) 
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            
        with tf.variable_scope('fc_2', reuse=reuse) as scope:
            c_weight_2 = _variable_with_weight_decay(
                'c_weight_2', 
                shape=[setting.FLAGS.final_bias, setting.FLAGS.num_classes], 
                stddev=0.01, 
                wd=0.005)
            c_bias_2 = tf.get_variable(
                'c_bias_2', 
                shape=[setting.FLAGS.num_classes], 
                initializer=tf.constant_initializer(0.0)) 
                
        with tf.variable_scope('softmax') as scope:     
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop , c_weight_2) + c_bias_2)
            
    return y_conv