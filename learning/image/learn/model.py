#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from setting import FLAGS

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

        def _activation_summary(x):
            tensor_name = x.op.name
            tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

        while i <= FLAGS.conv_loops:
            if i == 1:
                channels = FLAGS.channels
                biases_shape = FLAGS.bias
                conv_target = tf.reshape(image_batch_flatten, 
                    [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.channels]
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
                    padding='SAME', 
                    name='VALID_' + str(i))
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
                    padding='SAME',
                    name='VALID_' + str(i)))
            i += 1

        with tf.variable_scope('fc_1', reuse=reuse) as scope:
            dim = 1
            for d in pools[-1].get_shape()[1:].as_list():
                dim *= d
            
            c_weight_1 = _variable_with_weight_decay(
                'c_weight_1', 
                shape=[dim, FLAGS.final_bias], 
                stddev=0.01, 
                wd=0.005)
            c_bias_1 = tf.get_variable(
                'c_bias_1', 
                shape=[FLAGS.final_bias], 
                initializer=tf.constant_initializer(0.0))  
            pool_flatten = tf.reshape(pools[-1], [FLAGS.batch_size, dim])
            h_fc1 = tf.nn.relu(tf.matmul(pool_flatten, c_weight_1) + c_bias_1) 
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            
        with tf.variable_scope('fc_2', reuse=reuse) as scope:
            c_weight_2 = _variable_with_weight_decay(
                'c_weight_2', 
                shape=[FLAGS.final_bias, FLAGS.num_classes], 
                stddev=0.01, 
                wd=0.005)
            c_bias_2 = tf.get_variable(
                'c_bias_2', 
                shape=[FLAGS.num_classes], 
                initializer=tf.constant_initializer(0.0)) 
                
        with tf.variable_scope('softmax') as scope:     
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop , c_weight_2) + c_bias_2)
            
    return y_conv, c_weight_2, c_bias_2

def loss(labels, logits):
    '''
    calcute the difference between the inference and the correct label
    Args:
        labels = [batch_size,] tensor with correct label
        logits = inference with [batch_size, dim] tensor
    Return:
        loss = tensor object which shows the loss
    '''
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.summary.scalar('loss', loss)

    return loss

def training(total_loss, global_step=0, moving_average_decay=0.9999):
    '''
    train the data by optiziming
    Args:
        total_loss = tensor object which shows the loss
        global_step = 0
        moving_average_decay = 0.9999
    Return:
        optimized = optimized score
    '''
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        
    # Apply gradients, and add histograms
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(0.01)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
            
    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        optimized = tf.no_op(name='train')

    return optimized

def accuracy(logits, labels):
    '''
    calculate the accuracy of tthe learning
    Args:
        logits = tensor
        labels = tensor
    Return:
        accuracy = tensor
    '''
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.to_int64(labels))
        accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
        tf.summary.scalar('accuracy', accuracy)
        
    return accuracy