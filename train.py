import os
import sys
import time
import logging
import argparse
import numpy as np
import tensorflow as tf
from dispnet import DispNet
from util import init_logger, ft3d_filenames
from tensorflow.python.client import timeline

CORR = True


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
    parser.add_argument("-s", "--save_step", dest="save_step", type=int, default=5000,
                        help='save checkpoint step size')
    parser.add_argument("-n", "--n_steps", dest="n_steps", type=int, default=None,
                        help='test steps')
    parser.add_argument("--corr_type", dest="corr_type", type=str, default="tf",
                        help="correlation layer realization - 'tf' or 'cuda'")

    args = parser.parse_args()
    
    ft3d_dataset = ft3d_filenames(args.dataset_path)

    tf.logging.set_verbosity(tf.logging.ERROR)
    dispnet = DispNet(mode="traintest", ckpt_path=args.checkpoint_path, dataset=ft3d_dataset,
                      batch_size=args.batch_size, is_corr=CORR, corr_type="cuda")

    ckpt = tf.train.latest_checkpoint(args.checkpoint_path)
    if not ckpt:
        if not os.path.exists(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)
    model_name = "DispNet"
    if CORR:
        model_name += "Corr1D"
    init_logger(args.checkpoint_path, name=model_name)
    writer = tf.summary.FileWriter(args.checkpoint_path)

    schedule_step = 50000
    weights_schedule = [[0., 0., 0., 0., .2, 1.],
                        [0., 0., 0., .2, 1., .5],
                        [0., 0., .2, 1., .5, 0.],
                        [0., .2, 1., .5, 0., 0.],
                        [.2, 1., .5, 0., 0., 0.],
                        [1., .5, 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0., 0.]]
    lr_schedule = [1e-4] * 5
    for i in range(20):
        lr_schedule.extend([(lr_schedule[-1] / 2.)] * 3)

    log_step = args.log_step
    save_step = args.save_step
    test_step = save_step
    N_test = 1000

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
                dispnet.saver.restore(sess=sess, save_path=ckpt)
                step = int(ckpt[len(os.path.join(args.checkpoint_path, model_name))+1:])
                logging.info("step: %d" % step)
            else:
                step = 0
            schedule_current = min(step / schedule_step, len(weights_schedule)-1)
            feed_dict[dispnet.loss_weights] = np.array(weights_schedule[schedule_current])
            feed_dict[dispnet.learning_rate] = lr_schedule[schedule_current]
            while step < 5e5:
                if step % schedule_step == 0:
                    schedule_current = min(step / schedule_step, len(weights_schedule)-1)
                    feed_dict[dispnet.loss_weights] = np.array(weights_schedule[schedule_current])
                    feed_dict[dispnet.learning_rate] = lr_schedule[schedule_current]
                    logging.info("iter: %d, switching weights:" % step)
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
                step += dispnet.batch_size
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
                    dispnet.saver.save(sess, os.path.join(args.checkpoint_path, MODEL_NAME),
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
