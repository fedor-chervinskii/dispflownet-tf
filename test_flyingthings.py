import dispnet

import cv2
import time
import argparse
import numpy as np
from util import readPFM, ft3d_filenames
import tensorflow as tf

FT3D_PATH = '../datasets/FlyingThings3D'
MEAN_VALUE = 100.
BATCH_SIZE = 4

def preprocess(left_img, right_img, target, orig_size, input_size):
    left_img = tf.image.convert_image_dtype(left_img, tf.float32)
    mean = MEAN_VALUE #tf.reduce_mean(left_img)
    orig_width = orig_size[1]
    width, height, n_channels = input_size
    left_img = left_img - mean
    right_img = tf.image.convert_image_dtype(right_img, tf.float32)
    right_img = right_img - mean
    left_img = tf.image.resize_bilinear(left_img[np.newaxis, :, :, :], [height, width])[0]
    right_img = tf.image.resize_bilinear(right_img[np.newaxis, :, :, :], [height, width])[0]
    target = tf.image.resize_nearest_neighbor(target[np.newaxis, :, :, np.newaxis], [height, width])[0]
    target = target * width / orig_width
    left_img.set_shape([height, width, n_channels])
    right_img.set_shape([height, width, n_channels])
    target.set_shape([height, width, 1])
    return left_img, right_img, target

def read_sample(filename_queue):
    filenames = filename_queue.dequeue()
    left_fn, right_fn, disp_fn = filenames[0], filenames[1], filenames[2]
    left_img = tf.image.decode_image(tf.read_file(left_fn))
    right_img = tf.image.decode_image(tf.read_file(right_fn))
    target = tf.py_func(lambda x: readPFM(x)[0], [disp_fn], tf.float32)
    return left_img, right_img, target

def input_pipeline(filenames, orig_size, input_size, batch_size, num_epochs=1):
    filename_queue = tf.train.input_producer(
        filenames, element_shape = [3], num_epochs=num_epochs, shuffle=True)
    left_img, right_img, target = read_sample(filename_queue)
    left_img, right_img, target = preprocess(left_img, right_img, target, orig_size, input_size)
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    left_img_batch, right_img_batch, target_batch = tf.train.shuffle_batch(
        [left_img, right_img, target], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return left_img_batch, right_img_batch, target_batch

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

    orig_size = (540, 960, 3)
    input_size = (384, 768, 3)
    batch_size = 1
    log_step = 1
    
    ft3d_samples_filenames = ft3d_filenames(args.dataset_path)

    if args.n_steps is None:
        N_test = len(ft3d_samples_filenames["TEST"])
    else:
        N_test = args.n_steps

    graph = tf.Graph()
        
    with tf.Session(graph=graph) as sess:
#        try:
        ckpt = tf.train.latest_checkpoint(args.checkpoint_path)
        if ckpt:
            print("Restoring from %s" % ckpt)
            with graph.as_default():
                saver = tf.train.import_meta_graph(ckpt + ".meta")
                saver.restore(sess=sess, save_path=ckpt)
            init = graph.as_graph_element("init")
            sess.run(init)
            left_img, right_img, target = input_pipeline(ft3d_samples_filenames["TEST"],
                                                         orig_size=orig_size,
                                                         input_size=input_size,
                                                         batch_size=BATCH_SIZE,
                                                         num_epochs=1)
            print(graph.as_graph_element("left_image_batch"))
            left_image_batch = 0
            left_image_batch, right_image_batch, target = left_img, right_img, target
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print("queue runners started")
            test_err = 0
            print("Testing...")
            start = time.time()
            error = graph.get_tensor_by_name("error:0")
            training_mode = graph.get_tensor_by_name("training_mode:0")
            for i in range(N_test):
                err = sess.run([error], feed_dict={training_mode: False})
                test_err += err[0]
                if i % args.log_step == 0:
                    print("%d iterations, average pass in %f sec" % (i, (time.time()-start)/float(args.log_step)))
                    start = time.time()
            test_err = test_err / float(N_test)
            print("Average test EPE: %f" % test_err)
        else:
            print("Havent't found checkpoint in %s" % args.checkpoint_path)
#        except tf.errors.OutOfRangeError:
#            print("OutOfRangeError on %i'th" % (step))

#       finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()       
