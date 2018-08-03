import tensorflow as tf
from util import readPFM
import numpy as np
import os

LEAKY_ALPHA = 0.1
MAX_DISP = 40
MEAN_VALUE = 100.
INPUT_SIZE = (320, 896, 3)

initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)

# REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# shift_corr_module = tf.load_op_library(os.path.join(REPO_DIR, 'user_ops/shift_corr.so'))


def correlation(x, y, max_disp, is_train=True):
    x = tf.pad(x, [[0, 0], [0, 0], [max_disp, max_disp], [0, 0]], "CONSTANT")
    y = tf.pad(y, [[0, 0], [0, 0], [max_disp, max_disp], [0, 0]], "CONSTANT")
    corr = shift_corr_module.shift_corr(x, y, max_disp=max_disp)

    if is_train:
        @tf.RegisterGradient("ShiftCorr")
        def _ShiftCorrOpGrad(op, grad):
            return shift_corr_module.shift_corr_grad(op.inputs[0], op.inputs[1], grad, max_disp=max_disp)

    return tf.transpose(corr, perm=[0, 2, 3, 1])


def correlation_map(x, y, max_disp):
    corr_tensors = []
    y_shape = tf.shape(y)
    y_feature = tf.pad(y,[[0,0],[0,0],[max_disp,max_disp],[0,0]])
    for i in range(-max_disp, max_disp+1,1):
        shifted = tf.slice(y_feature, [0, 0, i + max_disp, 0], [-1, y_shape[1], y_shape[2], -1])
        corr_tensors.append(tf.reduce_mean(shifted*x, axis=-1, keepdims=True))

    result = tf.concat(corr_tensors,axis=-1)
    return result


def preprocess(left_img, right_img, target_img, conf_img, input_size, augmentation=False, conf_th=0):
    left_img = tf.image.convert_image_dtype(left_img, tf.float32)
    right_img = tf.image.convert_image_dtype(right_img, tf.float32)
    conf_img = tf.image.convert_image_dtype(conf_img, tf.float32)
    target_img = tf.cast(target_img,tf.float32)
    
    height, width, n_channels = input_size
    orig_width = tf.shape(left_img)[1]
    orig_height = tf.shape(left_img)[0]
    left_img = left_img - (MEAN_VALUE / 255)
    right_img = right_img - (MEAN_VALUE / 255)

    crop_row = tf.random_uniform(
        shape=(), minval=0, maxval=orig_height - height, dtype=tf.int32)
    crop_col = tf.random_uniform(
        shape=(), minval=0, maxval=orig_width - width, dtype=tf.int32)

    # random crop
    left_img = left_img[crop_row:crop_row + height, crop_col:crop_col + width, :]
    right_img = right_img[crop_row:crop_row + height, crop_col:crop_col + width, :]
    target = target_img[crop_row:crop_row + height, crop_col:crop_col + width, :]
    conf = conf_img[crop_row:crop_row + height, crop_col:crop_col + width, :]

    left_img.set_shape([height, width, n_channels])
    right_img.set_shape([height, width, n_channels])
    target = tf.reshape(target[:, :, 0], [height, width, 1])
    conf = tf.reshape(conf[:, :, 0], [height, width, 1])

    # mask out value below confidence 
    conf = tf.where(conf > conf_th, conf, tf.zeros_like(conf))

    # target should be multiplied by -1?
    target = -target

    if augmentation:
        active = tf.random_uniform(
            shape=[5], minval=0, maxval=1, dtype=tf.float32)
        # random gamma
        # random_gamma = tf.random_uniform(shape=(),minval=0.95,maxval=1.05,dtype=tf.float32)
        # left_img = tf.where(active[0]>0.5,left_img,tf.image.adjust_gamma(left_img,random_gamma))
        # right_img = tf.where(active[0]>0.5,right_img,tf.image.adjust_gamma(right_img,random_gamma))

        # random brightness
        random_delta = tf.random_uniform(
            shape=(), minval=-0.05, maxval=0.05, dtype=tf.float32)
        left_img = tf.where(
            active[1] > 0.5, left_img, tf.image.adjust_brightness(left_img, random_delta))
        right_img = tf.where(
            active[1] > 0.5, right_img, tf.image.adjust_brightness(right_img, random_delta))

        # random contrast
        random_contrast = tf.random_uniform(
            shape=(), minval=0.8, maxval=1.2, dtype=tf.float32)
        left_img = tf.where(active[2] > 0.5, left_img, tf.image.adjust_contrast(
            left_img, random_contrast))
        right_img = tf.where(active[2] > 0.5, right_img, tf.image.adjust_contrast(
            right_img, random_contrast))

        # random hue
        random_hue = tf.random_uniform(
            shape=(), minval=0.8, maxval=1.2, dtype=tf.float32)
        left_img = tf.where(active[3] > 0.5, left_img,
                            tf.image.adjust_hue(left_img, random_hue))
        right_img = tf.where(
            active[3] > 0.5, right_img, tf.image.adjust_hue(right_img, random_hue))

        # random_flip_left_right --> swap left and right image if they are flipped
        # temp = left_img
        # left_img = tf.where(active[4]>0.5,left_img,tf.image.flip_left_right(right_img))
        # right_img = tf.where(active[4]>0.5,right_img,tf.image.flip_left_right(temp))
        # target = tf.where(active[4]>0.5,target,tf.flip_left_right(target))

    left_img = tf.clip_by_value(left_img, -1, 1)
    right_img = tf.clip_by_value(right_img, -1, 1)

    return left_img, right_img, target, conf


def read_sample(filename_queue, pfm_target=True, scaled_gt=False, scaledConf=False):
    filenames = filename_queue.dequeue()
    left_fn, right_fn, disp_fn, conf_fn = filenames[0], filenames[1], filenames[2], filenames[3]
    left_img = tf.image.decode_image(tf.read_file(left_fn))
    right_img = tf.image.decode_image(tf.read_file(right_fn))
    if pfm_target:
        target = tf.py_func(lambda x: readPFM(x)[0], [disp_fn], tf.float32)
    else:
        read_type = tf.uint16 if scaled_gt else tf.uint8 
        target = tf.image.decode_png(tf.read_file(disp_fn),dtype=read_type)
        if scaled_gt:
            target = tf.cast(target,tf.float32)
            target = target/256.0
    
    read_type = tf.uint16 if scaledConf else tf.uint8 
    conf = tf.image.decode_png(tf.read_file(conf_fn),dtype=read_type)
    if scaledConf:
        conf = tf.image.convert_image_dtype(conf,tf.float32)
    return left_img, right_img, target, conf


def input_pipeline(filenames, input_size, batch_size, num_epochs=None, pfm_target=True, train=True, conf_th=0, scaledGt=False, scaledConf=False):
    filename_queue = tf.train.input_producer(filenames, element_shape=[4], num_epochs=num_epochs, shuffle=True)
    left_img, right_img, target, conf = read_sample(filename_queue, pfm_target=pfm_target,scaled_gt=scaledGt, scaledConf=scaledConf)
    left_img, right_img, target, conf = preprocess(left_img, right_img, target, conf, input_size, augmentation=train, conf_th=conf_th)
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    left_img_batch, right_img_batch, target_batch, conf_batch = tf.train.shuffle_batch(
        [left_img, right_img, target, conf], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads=8)

    return left_img_batch, right_img_batch, target_batch, conf_batch


def conv2d(x, kernel_shape, strides=1, relu=True, padding='SAME'):
    W = tf.get_variable("weights", kernel_shape, initializer=initializer)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable(
        "bias", kernel_shape[3], initializer=tf.constant_initializer(0.0))
    with tf.name_scope("conv"):
        x = tf.nn.conv2d(
            x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        #tf.summary.histogram("W", W)
        #tf.summary.histogram("b", b)
        # if kernel_shape[2] == 3:
        #     x_min = tf.reduce_min(W)
        #     x_max = tf.reduce_max(W)
        #     kernel_0_to_1 = (W - x_min) / (x_max - x_min)
        #     kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
        #     tf.summary.image('filters', kernel_transposed, max_outputs=3)
        if relu:
            x = tf.maximum(LEAKY_ALPHA * x, x)
    return x


def conv2d_transpose(x, kernel_shape, strides=1, relu=True):
    W = tf.get_variable("weights", kernel_shape, initializer=initializer)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable(
        "bias", kernel_shape[2], initializer=tf.constant_initializer(0.0))
    #output_shape = [x.get_shape()[0].value,x.get_shape()[1].value * strides, x.get_shape()[2].value * strides, kernel_shape[2]]
    x_shape = tf.shape(x)
    output_shape = [x_shape[0],x_shape[1]*strides,x_shape[2]*strides,kernel_shape[2]]
    with tf.name_scope("deconv"):
        x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1],
                                   padding='SAME')
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.maximum(LEAKY_ALPHA * x, x)
    return x


def upsampling_block(bottom, skip_connection, input_channels, output_channels, skip_input_channels):
    with tf.variable_scope("deconv"):
        deconv = conv2d_transpose(
            bottom, [4, 4, output_channels, input_channels], strides=2)
    with tf.variable_scope("predict"):
        predict = conv2d(
            bottom, [3, 3, input_channels, 1], strides=1, relu=False)
        #tf.summary.histogram("predict", predict)
    with tf.variable_scope("up_predict"):
        upsampled_predict = conv2d_transpose(
            predict, [4, 4, 1, 1], strides=2, relu=False)
    with tf.variable_scope("concat"):
        concat = conv2d(tf.concat([skip_connection, deconv, upsampled_predict], axis=3),
                        [3, 3, output_channels +
                            skip_input_channels + 1, output_channels],
                        strides=1, relu=False)
    return concat, predict


def build_main_graph(left_image_batch, right_image_batch, is_corr=True, corr_type="tf", is_train=True):
    if is_corr:
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
            if corr_type == "tf":
                corr = correlation_map(conv2a, conv2b, max_disp=MAX_DISP)
            else:
                corr = correlation(
                    conv2a, conv2b, max_disp=MAX_DISP, is_train=is_train)
        with tf.variable_scope("conv3"):
            conv3 = conv2d(tf.concat([corr, conv_redir], axis=3),
                           [5, 5, MAX_DISP * 2 + 1 + 64, 256], strides=2)
    else:
        with tf.variable_scope("conv1") as scope:
            conv1 = conv2d(tf.concat([left_image_batch, right_image_batch], axis=3), [
                           7, 7, 6, 64], strides=2)
        with tf.variable_scope("conv2") as scope:
            conv2 = conv2d(conv1, [5, 5, 64, 128], strides=2)
        with tf.variable_scope("conv3"):
            conv3 = conv2d(conv2, [5, 5, 128, 256], strides=2)
    with tf.variable_scope("conv3"):
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
        if is_corr:
            concat2, predict3 = upsampling_block(concat3, conv2a, 128, 64, 128)
        else:
            concat2, predict3 = upsampling_block(concat3, conv2, 128, 64, 128)
    with tf.variable_scope("up1"):
        if is_corr:
            concat1, predict2 = upsampling_block(concat2, conv1a, 64, 32, 64)
        else:
            concat1, predict2 = upsampling_block(concat2, conv1, 64, 32, 64)
    with tf.variable_scope("prediction"):
        predict1 = conv2d(concat1, [3, 3, 32, 1], strides=1, relu=False)
    return (predict1, predict2, predict3,
            predict4, predict5, predict6)


def L1_loss(gt, prediction, conf=None):
    #gt 0 means no gt
    abs_err = tf.abs(gt-prediction)
    if conf is None:
        valid_map = tf.where(tf.equal(gt,0), tf.zeros_like(gt, dtype=tf.float32), tf.ones_like(gt, dtype=tf.float32))
        filtered_error = abs_err*valid_map
    else:
        valid_map = tf.where(tf.equal(conf,0), tf.zeros_like(conf, dtype=tf.float32), tf.ones_like(conf, dtype=tf.float32))
        filtered_error = abs_err*conf
    return tf.reduce_sum(filtered_error)/tf.reduce_sum(valid_map) 


def build_loss(predictions, target, loss_weights, weight_decay, conf_batch=[], smoothness_lambda=0):
    height, width = target.get_shape()[1].value, target.get_shape()[2].value
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with tf.name_scope("loss"):
        targets = [tf.image.resize_nearest_neighbor(target, [height // np.power(2, n), width // np.power(2, n)]) for n in range(1, 7)]
        confs = [tf.image.resize_nearest_neighbor(conf_batch, [height // np.power(2, n), width // np.power(2, n)]) for n in range(1, 7)]
        if len(confs) == 0:
            losses = [L1_loss(targets[i], predictions[i]) for i in range(6)]
        else:
            losses = [L1_loss(targets[i], predictions[i], confs[i]) for i in range(6)]
        for i in range(6):
            tf.summary.scalar('loss' + str(i), losses[i])
            tf.summary.scalar('loss_weight' + str(i), loss_weights[i])

        # smoothness
        final_prediction = predictions[5]
        _, p_height, p_width, _ = final_prediction.shape
        diff_vert = tf.reduce_mean(tf.abs(final_prediction[:, 0:p_height - 1, :, :] - final_prediction[:, 1:, :, :]))
        diff_hor = tf.reduce_mean(tf.abs(final_prediction[:, :, 0:p_width - 1, :] - final_prediction[:, :, 1:, :]))
        mean_error = diff_vert * 2 + diff_hor * 2

        loss = tf.add_n([losses[i] * loss_weights[i] for i in range(6)])
        reg_loss = tf.contrib.layers.apply_regularization(regularizer)
        total_loss = loss + reg_loss + (smoothness_lambda * mean_error)
        tf.summary.scalar('loss', loss)
        error = losses[0]
        return total_loss, loss, error


class DispNet(object):
    def __init__(self, mode="inference", ckpt_path=".", dataset=None, input_size=INPUT_SIZE, batch_size=4, is_corr=True, corr_type="tf", smoothness_lambda=0, confidence_th=0, image_ops=None):
        self.ckpt_path = ckpt_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.is_corr = is_corr
        self.corr_type = corr_type
        self.dataset = dataset
        self.mode = mode
        self.smoothness_lambda = smoothness_lambda
        self.confidence_th = confidence_th
        self.image_ops = image_ops
        self.create_graph()

    def create_graph(self):
        self.loss_weights = tf.placeholder(
            tf.float32, shape=(6), name="loss_weights")
        self.learning_rate = tf.placeholder(
            tf.float32, shape=(), name="learning_rate")
        weight_decay = tf.placeholder_with_default(
            shape=(), name='weight_decay', input=0.0004)
        beta1 = tf.placeholder_with_default(shape=(), name="beta1", input=0.9)
        beta2 = tf.placeholder_with_default(shape=(), name="beta2", input=0.99)

        if self.mode == "traintest":
            train_pipeline = input_pipeline(self.dataset["TRAIN"], input_size=self.input_size, batch_size=self.batch_size, pfm_target=self.dataset['PFM'], train=True, conf_th=self.confidence_th, scaledGt=self.dataset['kitti_gt'], scaledConf=self.dataset['16bit_conf'])
            val_pipeline = input_pipeline(self.dataset["TEST"], input_size=self.input_size,batch_size=self.batch_size, pfm_target=self.dataset['PFM'], train=False, conf_th=self.confidence_th, scaledGt = self.dataset['kitti_gt'], scaledConf=self.dataset['16bit_conf'])

            with tf.variable_scope('model') as scope:
                left_image_batch, right_image_batch, target, conf_batch = train_pipeline
                self.predictions_train = build_main_graph(
                    left_image_batch, right_image_batch, is_corr=self.is_corr, corr_type=self.corr_type)

                scope.reuse_variables()

                left_image_test_batch, right_image_test_batch, target_test, _ = val_pipeline
                self.predictions_test = build_main_graph(
                    left_image_test_batch, right_image_test_batch, is_corr=self.is_corr, corr_type=self.corr_type, is_train=False)

            self.total_loss, self.loss, self.train_error = build_loss(self.predictions_train, target,
                                                                      self.loss_weights,
                                                                      weight_decay, conf_batch=conf_batch, smoothness_lambda=self.smoothness_lambda)
            # validation error
            target_rescaled = tf.image.resize_nearest_neighbor(
                target_test, [target_test.shape[1].value // 2, target_test.shape[2].value // 2])
            self.test_error = tf.reduce_mean(
                tf.abs(self.predictions_test[0] - target_rescaled))

            # summary ops
            tf.summary.scalar('train_error', self.train_error)
            tf.summary.image("left", left_image_batch, max_outputs=1)
            tf.summary.image("right", right_image_batch, max_outputs=1)
            for i in range(6):
                tf.summary.image(
                    "disp" + str(i), self.predictions_train[i], max_outputs=1)
            tf.summary.image("disp0_gt", target, max_outputs=1)
            tf.summary.image("conf0", conf_batch, max_outputs=1)

            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=beta1, beta2=beta2)
            self.train_step = optimizer.minimize(self.total_loss)
            self.mean_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('mean_loss', self.mean_loss)

        elif self.mode == "test":
            test_pipeline = input_pipeline(self.dataset["TEST"], input_size=self.input_size, batch_size=self.batch_size, pfm_target=self.dataset['PFM'], train=False, scaledGt=self.dataset['kitti_gt'], scaledConf=self.dataset['16bit_conf'])
            with tf.scope('model') as scope:
                left_image_batch, right_image_batch, target, _ = test_pipeline
                self.predictions_test = build_main_graph(
                    left_image_batch, right_image_batch, is_corr=self.is_corr, corr_type=self.corr_type)
        elif self.mode == "inference":
            assert self.image_ops is not None
            with tf.variable_scope('model') as scope:
                self.predictions_test = build_main_graph(
                    self.image_ops[0], self.image_ops[1], is_corr=self.is_corr, corr_type=self.corr_type)

        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())

        self.test_error = tf.placeholder(tf.float32)
        tf.summary.scalar('test_error', self.test_error)
        self.merged_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=2)
