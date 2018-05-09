import os
from os.path import join, relpath
from glob import glob
from PIL import Image
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import multiprocessing as mp
import concurrent.futures

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('directory', '../../../data/tfrecord', """Directory where to write *.tfrecords.""")

NUMBER_OF_TFREOCRDS = 1
MAX_WORKER = 10
LABELS = [
    ['ル・クルーゼ', 0],
    ['ストウブ',    1],
    ['シャスール',   2],
    ['バーミキュラ', 3]
]

def chunked(iterable, n):
    return [iterable[x:x + n] for x in range(0, len(iterable), n)]

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(dataset, name='train'):
    '''
    Please check below url if error
    however there is continue when error occurs
    https://stackoverflow.com/questions/48944819/image-open-gives-error-cannot-identify-image-file
    '''
    div = round(len(dataset) / NUMBER_OF_TFREOCRDS)
    chunk = chunked(dataset, div)
    
    i = 1
    for list in chunk: 
        filename = os.path.join(FLAGS.directory, name + '_' + str(i) + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)

        for data in list:
            try:
                print(data[0])
                
                image_path = data[0]
                label = int(data[1])

                image_object = Image.open(image_path)
                path, ext = os.path.splitext(image_path)
                image = np.array(image_object)

                height = image.shape[0]
                width = image.shape[1]
                depth = 3
                image = image.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(label),
                    'path' : _bytes_feature(image_path.encode('utf-8'))}))
                writer.write(example.SerializeToString())
            except:
                continue

        writer.close()
        i += 1


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.directory):
        tf.gfile.MakeDirs(FLAGS.directory)

    img_data = []
    for n, v in LABELS:
        path = os.path.join("../../../data/image", n)
        
        for file in [relpath(x, path) for x in glob(join(path, '*.jpg'))]:
            img_data.append([os.path.join(path, file), v])
            
        for file in [relpath(x, path) for x in glob(join(path, '*.jpeg'))]:
            img_data.append([os.path.join(path, file), v])
            
        for file in [relpath(x, path) for x in glob(join(path, '*.png'))]:
            img_data.append([os.path.join(path, file), v])

    #p = Pool(mp.cpu_count())
    executor = concurrent.futures.ThreadPoolExecutor(max_workers = MAX_WORKER)
    executor.submit(convert_to(img_data))
    #p.map(convert_to(img_data))
    #p.close()

if __name__ == '__main__':
    tf.app.run()