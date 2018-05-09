#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from . import setting
from . import helper
import numpy as np
from . import model

def predictUsingPb(image_path, pb_path):
    '''
    Calculate and returns the prediction using pb file
    Args:
        image_path = full path to the image
        pb_path = full path to the pb file
    Return:
        prediction = interger
    '''
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            input = helper._getImage(image_path)
            input = tf.reshape(input, [-1, setting.IMAGE_PIXELS])

            prediction = np.argmax(sess.run(
                'output:0',
                {'input:0' : input.eval()}
            ))
            
    return prediction

def predictUsingCkpt(image_path, ckpt_path):
    '''
    Calculate and returns the prediction using ckpt file
    Args:
        image_path = full path to the image
        ckpt_path = full path to the ckpt file
    Return:
        prediction = interger
    '''
    graph = tf.Graph()
    prediction = None
    with graph.as_default():
        image_placeholder = tf.placeholder(tf.float32, shape=[None, setting.IMAGE_PIXELS])
        keep_prob = tf.placeholder(tf.float32)

        input = helper._getImage(image_path)
        input = tf.reshape(input, [-1, setting.IMAGE_PIXELS])

        logits = model.inference(
            image_placeholder, keep_prob=keep_prob
        )
    
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            if ckpt is not None:
                saver = tf.train.Saver(tf.global_variables())
                last_model = ckpt.model_checkpoint_path
                saver.restore(sess, last_model)
                
                prediction = np.argmax(logits.eval(feed_dict={
                    image_placeholder : input.eval(),
                    keep_prob : 1.0
                })[0])
    
    return prediction
    