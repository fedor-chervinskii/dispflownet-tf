import os
import cv2
import sys
import time
import logging
import datetime
import argparse
import numpy as np
import tensorflow as tf
from dispnet import DispNet
from util import readPFM, ft3d_filenames
from tensorflow.python.client import timeline

INPUT_SIZE = (384, 768, 3)
MODEL_NAME = "DispNetCorr1D"


def init_logger(log_path):
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)
    logfile = os.path.join(log_path, "dispnet-%s.log" % datetime.datetime.today())
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setLevel(logging.INFO)
    root.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.terminator = ""
    root.addHandler(consoleHandler)
    logging.debug("Logging to %s" % logfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="dataset_path", required=True, type=str,
                        metavar="FILE", help='path to FlyingThings3D dataset')
    parser.add_argument("-c", "--ckpt", dest="checkpoint_path", default=".", type=str,
                        metavar="FILE", help='model checkpoint path')
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=4, type=int,
                        help='batch size')
    parser.add_argument("-l", "--log_step", dest="log_step", type=int, default=100,
                        help='log step size')
    parser.add_argument("-s", "--save_step", dest="save_step", type=int, default=2000,
                        help='save checkpoint step size')
    parser.add_argument("-n", "--n_steps", dest="n_steps", type=int, default=None,
                        help='test steps')

    args = parser.parse_args()
    
    ft3d_dataset = ft3d_filenames(args.dataset_path)

    tf.logging.set_verbosity(tf.logging.ERROR)
    dispnet = DispNet(mode="traintest", ckpt_path=args.checkpoint_path, dataset=ft3d_dataset,
                      input_size=INPUT_SIZE, batch_size=args.batch_size, corr_type="cuda")

    ckpt = tf.train.latest_checkpoint(args.checkpoint_path)
    if not ckpt:
        if not os.path.exists(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)
    init_logger(args.checkpoint_path)
    writer = tf.summary.FileWriter(args.checkpoint_path)

    schedule_step = 10000
    weights_schedule = [[0., 0., 0., 0., .2, 1.],
                        [0., 0., 0., .2, 1., .5],
                        [0., 0., .2, 1., .5, 0.],
                        [0., .2, 1., .5, 0., 0.],
                        [.2, 1., .5, 0., 0., 0.],
                        [1., .5, 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0., 0.]]
    lr_schedule = [1e-4] * 7
    for i in range(20):
        lr_schedule.extend([(lr_schedule[-1] / 2.)] * 4)

    log_step = args.log_step
    save_step = args.save_step
    test_step = save_step
    N_test = 250

    gpu_options = tf.GPUOptions(allow_growth=True)
#    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#    run_metadata = tf.RunMetadata()
    with tf.Session(graph=dispnet.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(dispnet.init)
        logging.debug("initialized")
        writer.add_graph(sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        logging.debug("queue runners started")
        try:
            feed_dict = {}
            feed_dict[dispnet.training_mode] = True
            l_mean = 0
            start = time.time()
            feed_dict[dispnet.test_error] = None
            ckpt = tf.train.latest_checkpoint(args.checkpoint_path)
            if ckpt:
                logging.info("Restoring from %s" % ckpt)
                saver.restore(sess=sess, save_path=ckpt)
                step = int(ckpt[len(os.path.join(args.checkpoint_path, MODEL_NAME))+1:])
                print("step: %d" % step)
            else:
                step = 0
            schedule_current = min(step / schedule_step, len(weights_schedule)-1)
            feed_dict[dispnet.loss_weights] = np.array(weights_schedule[schedule_current])
            feed_dict[dispnet.learning_rate] = lr_schedule[schedule_current]
            while step < 5e5:
                if step % schedule_step == 0:
                    feed_dict[dispnet.loss_weights] = np.array(weights_schedule[schedule_current])
                    feed_dict[dispnet.learning_rate] = lr_schedule[schedule_current]
                    logging.info("switching weights:")
                    logging.info(feed_dict[dispnet.loss_weights])
                    logging.info("learning rate: %f" % feed_dict[dispnet.learning_rate])
                _, l, err = sess.run([dispnet.train_step, dispnet.loss, dispnet.error],
                                      feed_dict=feed_dict)#, options=options, run_metadata=run_metadata)
                # trg, pred = sess.run([target, predictions], feed_dict=feed_dict)
#                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#                chrome_trace = fetched_timeline.generate_chrome_trace_format()
#                with open('timeline_%d.json' % step, 'w') as f:
#                    f.write(chrome_trace)
                l_mean += l
                step += 1
                if step % log_step == 0:
                    l_mean = np.array(l_mean / float(log_step))
                    feed_dict[dispnet.mean_loss] = l_mean
                    s = sess.run(dispnet.merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, step)
                    logging.debug("iter: %d, f/b pass time: %f, loss: %f, error %f" %
                                  (step, ((time.time() - start) / float(log_step)), l_mean, err))
                    l_mean = 0
                    start = time.time()
                if step % save_step == 0:
                    logging.info("saving to file %s." % 
                                 (os.path.join(args.checkpoint_path, MODEL_NAME)))
                    saver.save(sess, os.path.join(args.checkpoint_path, MODEL_NAME),
                               global_step=step)
                if step % test_step == 0:
                    test_err = 0
                    feed_dict[dispnet.training_mode] = False
                    logging.info("Testing...")
                    for j in range(N_test):
                        err = sess.run([dispnet.error], feed_dict=feed_dict)
                        test_err += err[0]
                    test_err = test_err / float(N_test)
                    logging.info("Test error %f" % test_err)
                    feed_dict[dispnet.test_error] = test_err

        except tf.errors.OutOfRangeError:
            logging.INFO('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))

        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()
