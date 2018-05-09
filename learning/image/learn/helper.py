#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import random
from glob import glob
from setting import FLAGS

def getTFrecords(dir_path):
    '''
    dir_path = directory path to the tfrecords
    return None/list of tfrecord paths
    '''
    #get all the tfrecords in the list format
    files = os.listdir(dir_path)
    if len(files) == 0:
        return None

    tfrecords = []
    for file in files:
        tfrecords.append(str(dir_path) + str(file))
        
    return tfrecords

def readTFrecord(path):
    '''
    read tfrecords from givin path
    '''
    filename_queue = tf.train.string_input_producer(path)
    reader = tf.TFRecordReader()
    _, serialized_data = reader.read(filename_queue, 'train')

    features = tf.parse_single_example(
        serialized_data,
        features={
            "label"       : tf.FixedLenFeature([], tf.int64),
            "path"        : tf.FixedLenFeature([], tf.string),
        })

    label = tf.cast(features['label'], tf.int32)
    path = features['path']

    return label, path

def image_from_path(path, channels=3, image_size=96, input_size=112):
    '''
    do not use gif.
    
    from givin path, it will output randmly created image with 
    Args:
        path = path obtained from tfrecord
        channels = 3
        image_size = 96
        input_size = 112
    Return:
        image = size of 96 x 96
    '''
    img_bytes = tf.read_file(path)
    image = tf.image.decode_jpeg(img_bytes, channels=channels)

    cropsize = random.randint(image_size, image_size + (input_size - image_size) / 2)
    framesize = image_size + (cropsize - image_size) * 2
    image = tf.image.resize_image_with_crop_or_pad(image, framesize, framesize)
    image = tf.random_crop(image, [cropsize, cropsize, channels])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_hue(image, max_delta=0.04)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_images(image, [image_size, image_size])

    return image

def batched(label, path, batch_size, channels=3, image_size=96):
    '''
    this will output the label and images in the form of below
    Args:
        label = label number obtained from tfrecord
        path = path obtained from tfrecord
        channels = 3 
        batch_size = FLAGS.batch_size
        image_size = 96
    Return:
        label_batch [batch_size, num_classes]
        image_batch [batch_size, image_pixal]
    '''
    label_batch, path_batch = tf.train.shuffle_batch(
        [label, path],
        batch_size=1,
        min_after_dequeue=70000,
        capacity=70000 + 3 * 1,
        num_threads=1
    )

    label_batch_flatten = tf.reshape(label_batch, [-1])
    path_batch_flatten = tf.reshape(path_batch, [-1])
    image_batch_flatten = tf.map_fn(image_from_path, path_batch_flatten, dtype=tf.float32)

    label_batch, image_batch = tf.train.batch(
        [label_batch_flatten, image_batch_flatten],
        batch_size=batch_size,
        capacity=10000 + 3 * batch_size,
        allow_smaller_final_batch=True
    )

    label_batch = tf.reshape(label_batch, [-1])
    
    image_batch = tf.reshape(image_batch, [-1, image_size, image_size, channels])
    tf.summary.image('images', image_batch)
    
    image_pixal = image_size * image_size * channels
    image_batch = tf.reshape(image_batch, [-1, image_pixal])

    return label_batch, image_batch

def _getImage(image_path):
    '''
    Obtain and return an image and return in numpy array format
    Args:
        image_path = full path to the image
        image_size = 96
        channels = 3
    Returns:
        tf image
    '''
    
    img_bytes = tf.read_file(image_path)
    image = tf.image.decode_jpeg(img_bytes, channels=FLAGS.channels)
    image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.image_size, FLAGS.image_size)
    
    return image
    
def get_latest_modified_file_path(dirname):
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
    return latest_modified_file_path[0]