import dispnet

import os
import cv2
import glob
import time
import argparse
import numpy as np
import tensorflow as tf

FT3D_PATH = '../datasets/FlyingThings3D'

def ft3d_filenames(path):
    ft3d_path = path
    ft3d_samples_filenames = {}
    for prefix in ["TRAIN", "TEST"]:
        ft3d_train_data_path = os.path.join(ft3d_path, 'frames_cleanpass/TRAIN')
        ft3d_train_labels_path = os.path.join(ft3d_path, 'disparity/TRAIN')
        left_images_filenames = sorted(glob.glob(ft3d_train_data_path + "/*/*/left/*"))
        right_images_filenames = sorted(glob.glob(ft3d_train_data_path + "/*/*/right/*"))
        disparity_filenames = sorted(glob.glob(ft3d_train_labels_path + "/*/*/left/*"))

        ft3d_samples_filenames[prefix] = [(left_images_filenames[i],
                                           right_images_filenames[i],
                                           disparity_filenames[i]) for i in range(len(left_images_filenames))]
    return ft3d_samples_filenames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="dataset_path", required=True, type=str,
            metavar="FILE", help='path to FlyingThings3D dataset')
    parser.add_argument("-c", "--ckpt", dest="checkpoint_path", required=True, type=str,
            metavar="FILE", help='model checkpoint path')
    parser.add_argument("-l", "--log_step", dest="log_step", type=int, default=100,
            help='log step size')
    parser.add_argument("-n", "--n_steps", dest="n_steps", type=int, default=None,
            help='test steps')

    args = parser.parse_args()
    
    tf.reset_default_graph()

    graph = tf.Graph()

    orig_size = (540, 960, 3)
    input_size = (384, 768, 3)
    batch_size = 1
    log_step = 1
    
    ft3d_samples_filenames = ft3d_filenames(args.dataset_path)

    if args.n_steps is None:
        N_test = len(ft3d_samples_filenames["TEST"])
    else:
        N_test = args.n_steps

    with graph.as_default():

        loss_weights = tf.placeholder(tf.float32, shape=(6),
                                      name="loss_weights")
        learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        weight_decay = tf.placeholder_with_default(shape=(), name="weight_decay", input=0.0004)
        beta1 = tf.placeholder_with_default(shape=(), name="beta1", input=0.9)
        beta2 = tf.placeholder_with_default(shape=(), name="beta2", input=0.99)

        training_mode = tf.placeholder_with_default(shape=(), input=False)

        train_pipeline = dispnet.input_pipeline(ft3d_samples_filenames["TRAIN"], orig_size=orig_size,
                                                input_size=input_size, batch_size=batch_size, num_epochs=None)

        val_pipeline = dispnet.input_pipeline(ft3d_samples_filenames["TEST"], orig_size=orig_size,
                                              input_size=input_size, batch_size=batch_size, num_epochs=None)

        left_image_batch, right_image_batch, target = tf.cond(training_mode,
                                                              lambda: train_pipeline,
                                                              lambda: val_pipeline)    

        predictions = dispnet.build_main_graph(left_image_batch, right_image_batch)
        total_loss, loss, error = dispnet.build_loss(predictions, target, loss_weights, weight_decay)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        train_step = optimizer.minimize(total_loss)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        mean_loss = tf.placeholder(tf.float32)
        tf.summary.scalar('mean_loss', mean_loss)

        merged_summary = tf.summary.merge_all()

        test_error = tf.placeholder(tf.float32)
        test_error_summary = tf.summary.scalar('test_error', test_error)

        saver = tf.train.Saver()
        
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        print("initialized")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("queue runners started")
        try:
            ckpt = tf.train.latest_checkpoint(args.checkpoint_path)
            if ckpt:
                print("Restoring from %s" % ckpt)
                saver.restore(sess=sess, save_path=ckpt)
                test_err = 0
                feed_dict = {}
                feed_dict[training_mode] = False
                print("Testing...")
                start = time.time()
                for i in range(N_test):
                    err = sess.run([error], feed_dict=feed_dict)
                    test_err += err[0]
                    if i % args.log_step == 0:
                        print("%d iterations, average pass in %f sec" % (i, (time.time()-start)/float(args.log_step)))
                        start = time.time()
                test_err = test_err / float(N_test)
                print("Average test EPE: %f" % test_err)
            else:
                print("Havent't found checkpoint in %s" % args.checkpoint_path)
        except tf.errors.OutOfRangeError:
            print("OutOfRangeError on %i'th" % (step))

        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()       
