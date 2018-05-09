#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from . import setting

def _getImage(image_path):
    '''
    Obtain and return an image and return in numpy array format
    Args:
        image_path = full path to the image
    Returns:
        tf image
    '''
    
    img_bytes = tf.read_file(image_path)
    image = tf.image.decode_jpeg(img_bytes, channels=setting.FLAGS.channels)
    image = tf.image.resize_image_with_crop_or_pad(image, setting.FLAGS.image_size, setting.FLAGS.image_size)
    
    return image