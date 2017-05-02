import tensorflow as tf
from util import readPFM
import numpy as np

LEAKY_ALPHA = 0.1
MAX_DISP=40

initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)

def preprocess(left_img, right_img, target, orig_size, input_size):
    left_img = tf.image.convert_image_dtype(left_img, tf.float32)
    mean = tf.reduce_mean(left_img)
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

def input_pipeline(filenames, orig_size, input_size, batch_size, num_epochs=None):
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

def correlation_map(x, y, max_disp):
    corr_tensors = []
    for i in range(-max_disp, 0, 1):
        shifted = tf.pad(tf.slice(y, [0, 0, -i, 0], [-1]*4),
                                  [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, y), axis=3)
        corr_tensors.append(corr)
    for i in range(max_disp + 1):
        shifted = tf.pad(tf.slice(y, [0]*4, [-1, -1, y.shape[2].value - i, -1]),
                                  [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, y), axis=3)
        corr_tensors.append(corr)
    return tf.transpose(tf.stack(corr_tensors),
                        perm=[1,2,3,0])

def conv2d(x, kernel_shape, strides=1, relu=True, padding='SAME'):
    W = tf.get_variable("weights", kernel_shape, initializer=initializer)
    b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))
    with tf.name_scope("conv"):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        tf.summary.histogram("W", W)
        tf.summary.histogram("b", b)
        if kernel_shape[2] == 3:
            x_min = tf.reduce_min(W)
            x_max = tf.reduce_max(W)
            kernel_0_to_1 = (W - x_min) / (x_max - x_min)
            kernel_transposed = tf.transpose (kernel_0_to_1, [3, 0, 1, 2])
            tf.summary.image('filters', kernel_transposed, max_outputs=3)       
        if relu : x = tf.maximum(LEAKY_ALPHA * x, x)
    return x

def conv2d_transpose(x, kernel_shape, strides=1, relu=True):
    W = tf.get_variable("weights", kernel_shape, initializer=initializer)
    b = tf.get_variable("biases", kernel_shape[2], initializer=tf.constant_initializer(0.0))
    output_shape = [x.get_shape()[0].value,
                    x.get_shape()[1].value*strides, x.get_shape()[2].value*strides, kernel_shape[2]]
    with tf.name_scope("deconv"):
        x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1],
                                   padding='SAME')
        x = tf.nn.bias_add(x, b)
        if relu : x = tf.maximum(LEAKY_ALPHA * x, x)
    return x

def upsampling_block(bottom, skip_connection, input_channels, output_channels, skip_input_channels):
    with tf.variable_scope("deconv"):
        deconv = conv2d_transpose(bottom, [4, 4, output_channels, input_channels], strides=2)
    with tf.variable_scope("predict"):
        predict = conv2d(bottom, [3, 3, input_channels, 1], strides=1, relu=False)
        tf.summary.histogram("predict", predict)
    with tf.variable_scope("up_predict"):
        upsampled_predict = conv2d_transpose(predict, [4, 4, 1, 1], strides=2, relu=False)
    with tf.variable_scope("concat"):
        concat = conv2d(tf.concat([skip_connection, deconv, upsampled_predict], axis=3),
                        [3, 3, output_channels + skip_input_channels + 1, output_channels],
                        strides=1, relu=False)
    return concat, predict
    

def build_main_graph(left_image_batch, right_image_batch):
    with tf.variable_scope("conv1") as scope:
        conv1a = conv2d(left_image_batch, [7, 7, 3, 64], strides=2)
        scope.reuse_variables()
        conv1b = conv2d(right_image_batch, [7, 7, 3, 64], strides=2)
    with tf.variable_scope("conv2") as scope:
        conv2a = conv2d(conv1a, [5, 5, 64, 128], strides=2)
        scope.reuse_variables()
        conv2b = conv2d(conv1b, [5, 5, 64, 128], strides=2)
    with tf.variable_scope("conv_redir"):
        conv_redir = conv2d(conv2a, [1, 1, 128, 64], strides=1)
    with tf.name_scope("correlation"):
        corr = correlation_map(conv2a, conv2b, max_disp=MAX_DISP)
#        corr = correlation(conv2a, conv2b, max_disp=MAX_DISP)
    with tf.variable_scope("conv3"):
        conv3 = conv2d(tf.concat([corr, conv_redir], axis=3), [5, 5, 145, 256], strides=2)
        with tf.variable_scope("1"):
            conv3_1 = conv2d(conv3, [3, 3, 256, 256], strides=1)
    with tf.variable_scope("conv4"):
        conv4 = conv2d(conv3_1, [3, 3, 256, 512], strides=2)
        with tf.variable_scope("1"):
            conv4_1 = conv2d(conv4, [3, 3, 512, 512], strides=1)
    with tf.variable_scope("conv5"):
        conv5 = conv2d(conv4_1, [3, 3, 512, 512], strides=2)
        with tf.variable_scope("1"):
            conv5_1 = conv2d(conv5, [3, 3, 512, 512], strides=1)
    with tf.variable_scope("conv6"):
        conv6 = conv2d(conv5_1, [3, 3, 512, 1024], strides=2)
        with tf.variable_scope("1"):
            conv6_1 = conv2d(conv6, [3, 3, 1024, 1024], strides=1)
    with tf.variable_scope("up5"):
        concat5, predict6 = upsampling_block(conv6_1, conv5_1, 1024, 512, 512)
    with tf.variable_scope("up4"):
        concat4, predict5 = upsampling_block(concat5, conv4_1, 512, 256, 512)
    with tf.variable_scope("up3"):
        concat3, predict4 = upsampling_block(concat4, conv3_1, 256, 128, 256)
    with tf.variable_scope("up2"):
        concat2, predict3 = upsampling_block(concat3, conv2a, 128, 64, 128)
    with tf.variable_scope("up1"):
        concat1, predict2 = upsampling_block(concat2, conv1a, 64, 32, 64)
    with tf.variable_scope("prediction"):
        predict1 = conv2d(concat1, [3, 3, 32, 1], strides=1, relu=False)
    return (predict1, predict2, predict3,
            predict4, predict5, predict6)

def L1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))

def build_loss(predictions, target, loss_weights):
    height, width = target.get_shape()[1].value, target.get_shape()[2].value
    with tf.name_scope("loss"):
        targets = [tf.image.resize_nearest_neighbor(target, [height / np.power(2, n),
                                                         width / np.power(2, n)])
                   for n in range(1, 7)]
        losses = [L1_loss(targets[i], predictions[i]) for i in range(6)]
        for i in range(6):
            tf.summary.scalar('loss' + str(i), losses[i])
            tf.summary.scalar('loss_weight' + str(i), loss_weights[i])
        loss = tf.add_n([losses[i]*loss_weights[i] for i in range(6)])
        tf.summary.scalar('loss', loss)
        error = losses[0]
        return loss, error