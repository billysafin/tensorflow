#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from setting import FLAGS
from setting import IMAGE_PIXELS
from setting import LABELS
import model
import helper
import time
from datetime import datetime
import numpy as np
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug
import predictor

def main(argv=None):
    '''
    main process for running image recongization
    
    unless specified, channels will be 3 (colored)
    '''
    graph = tf.Graph()
    with graph.as_default():
        #placeholders for image_batch and label batch
        image_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
        label_placeholder = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        keep_prob = tf.placeholder(tf.float32)
        
        #logits for learning
        logits, weight, bias = model.inference(
            image_placeholder, keep_prob=keep_prob
        )
        
        logits = tf.identity(logits, name='out_node')

        #loss for learning
        loss = model.loss(label_placeholder, logits)
        
        #optimazition for learning
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = model.training(loss, global_step)
        
        #accuracy
        acc = model.accuracy(logits, label_placeholder)
        
        #read tfrecords
        tfrecords = helper.getTFrecords(FLAGS.data_dir_mo)
        labels, image_paths = helper.readTFrecord(tfrecords)
        labels_batch, images_batch = helper.batched(labels, image_paths, FLAGS.batch_size)
        
        #creave saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
        
        #for tensor board and learned data
        tf.summary.FileWriterCache.clear()
        
        #performance
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
        #merge all graphs
        summary = tf.summary.merge_all()
        
        with tf.Session(graph=graph) as sess:
            #debugging
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            
            #check for checkpoint files
            if FLAGS.restudy_old_model == 1:
                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                if ckpt is not None:
                    last_model = ckpt.model_checkpoint_path
                    print("Loading Last saved Model: " + last_model)
                    saver.restore(sess, last_model)
                else: 
                    #initialize both global and local variables
                    sess.run([
                        tf.global_variables_initializer()
                        ,tf.local_variables_initializer()
                    ])
            else:
                #initialize both global and local variables
                sess.run([
                    tf.global_variables_initializer()
                    ,tf.local_variables_initializer()
                ]) 
            
            #write train summaries for tensorboard
            writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, sess.graph)

            #begin learning and print the loss for each steps
            #prepare threads to obtain data from batches
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            try:
                start_time_total = time.time()
                for step in range(1, FLAGS.max_steps + 1):
                    start_time = time.time()
                    
                    #batches
                    labels, images = sess.run([labels_batch, images_batch])
                    
                    _ = sess.run(
                        [train_op]
                        ,feed_dict = {
                            image_placeholder   : images
                            , label_placeholder : labels
                            , keep_prob         : 1.0
                        }
                    )

                    trained_accurcy, train_loss, summary_str = sess.run(
                        [acc, loss, summary]
                        , feed_dict = {
                            image_placeholder   : images
                            , label_placeholder : labels
                            , keep_prob         : 0.5
                        },
                        options=run_options,
                        run_metadata=run_metadata
                    )

                    writer.add_summary(summary_str, step)
                    writer.flush()

                    step_stats = run_metadata.step_stats
                    tl = timeline.Timeline(step_stats)

                    ctf = tl.generate_chrome_trace_format(show_memory=False,
                        show_dataflow=True)

                    with open(FLAGS.log_dir + "timeline.json", "w") as f:
                        f.write(ctf)

                    duration = time.time() - start_time
                    format_learning = '%s: step %d, loss = %.5f , accuracy %.5f (%.3f sec/batch)'
                    print(format_learning % (datetime.now(), step, train_loss, trained_accurcy, duration))

                    if step % 100 == 0:
                        save_path = saver.save(sess, FLAGS.checkpoint_dir + 'model', global_step=step)
                        print("The model is saved to the file: %s" % save_path)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                #print total time took to learn
                total_time = time.time() - start_time_total
                print('total time took to learn (%.3f sec)' % (total_time))
                
                #saving the final result
                save_path = saver.save(sess, FLAGS.checkpoint_dir + 'model', global_step=step)
                print("The model is saved to the file: %s" % save_path)
                
                #for creating pb file
                if FLAGS.save_as_pb == 1:
                    g_2 = tf.Graph()
                    with g_2.as_default():
                        x_2 = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name='input')
                        x_2 = tf.reshape(x_2, [-1, 1024])
                        W_2 = tf.constant(sess.run(weight), name='weight')
                        b_2 = tf.constant(sess.run(bias), name='bias')
                        y_2 = tf.nn.softmax(tf.matmul(x_2, W_2) + b_2, name='output')
                        sess_2 = tf.Session()
                        init_2 = tf.global_variables_initializer()
                        sess_2.run(init_2)
                        
                        graph_def = g_2.as_graph_def()
                        
                        date = datetime.now().strftime('%Y%m%d')
                        tf.train.write_graph(graph_def, FLAGS.checkpoint_dir, date + '.pb', as_text=False)
                        print('Pb file was save to and as ' + FLAGS.checkpoint_dir + date + '.pb')
                        
                coord.request_stop()
            coord.join(threads)
    
    #testing the checkpoint file accuracy
    graph_3 = tf.Graph()
    with graph_3.as_default():
        image_placeholder_3 = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
        keep_prob_3 = tf.placeholder(tf.float32)

        image_path = helper.get_latest_modified_file_path(FLAGS.test_image_dir)
        print("Checkpoint: using image is: " + image_path)
        input = helper._getImage(image_path)
        input = tf.reshape(input, [-1, IMAGE_PIXELS])

        logits_3 = predictor.pred_inference(
            image_placeholder_3, keep_prob=keep_prob_3
        )
    
        with tf.Session() as sess_3:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt is not None:
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
                last_model = ckpt.model_checkpoint_path
                print("Loading Last saved Model: " + last_model)
                saver.restore(sess_3, last_model)
                
                pred = np.argmax(logits_3.eval(feed_dict={
                    image_placeholder_3 : input.eval(),
                    keep_prob_3 : 1.0
                })[0])

                if pred in LABELS:
                    print("Predcition using checkpoint is " + str(LABELS[pred]))
                else:
                    print("failed to predict")
            else:
                print("Failed to load the last saved Model")
                
    #testing pb file
    if FLAGS.save_as_pb == 1:
        with tf.gfile.FastGFile(FLAGS.checkpoint_dir + date + '.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

            with tf.Session() as sess_4:
                image_path = helper.get_latest_modified_file_path(FLAGS.test_image_dir)
                print("Protocal Buff: using image is: " + image_path)
                input = helper._getImage(image_path)
                input = tf.reshape(input, [-1, IMAGE_PIXELS])

                prediction = np.argmax(sess_4.run(
                    'output:0',
                    {'input:0' : input.eval()}
                ))

                if pred in LABELS:
                    print("Predcition using pb is " + str(LABELS[prediction]))
                else:
                    print("failed to predict using pb file")
             
if __name__ == '__main__':
    tf.app.run()