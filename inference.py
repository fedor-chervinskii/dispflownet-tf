import argparse
import tensorflow as tf
import numpy as np
import os
from dispnet import DispNet
from util import get_var_to_restore_list
from matplotlib import pyplot as plt
import cv2

# INPUT_SIZE = (384, 768, 3)
# INPUT_SIZE = (540, 960, 3)
DOWNGRADE_FACTOR = 64

def pad_image(immy,down_factor = DOWNGRADE_FACTOR):
    """
    pad image with a proper number of 0 to prevent problem when concatenating after upconv
    """
    immy_shape = tf.shape(immy)
    new_height = tf.where(tf.equal(immy_shape[0]%down_factor,0),x=immy_shape[0],y=(tf.floordiv(immy_shape[0],down_factor)+1)*down_factor)
    new_width = tf.where(tf.equal(immy_shape[1]%down_factor,0),x=immy_shape[1],y=(tf.floordiv(immy_shape[1],down_factor)+1)*down_factor)
    immy = tf.image.resize_image_with_crop_or_pad(immy,new_height,new_width)
    return immy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=str, metavar="FILE",
                        help='path to the folder with left image or file with path to elaborate (one per row)')
    parser.add_argument("--right", required=True, type=str, metavar='FILE',
                        help="path to the folder with right image or file with path to elaborate (one per row)")
    parser.add_argument("-c", "--ckpt", dest="checkpoint_path",
                        default=".", help='model checkpoint path')
    parser.add_argument("--corr_type", dest="corr_type", type=str, default="tf",
                        help="correlation layer realization", choices=['tf', 'cuda', 'none'])
    parser.add_argument("-o", "--output", required=True,
                        help="path were the predictions will be saved")
    parser.add_argument("-v", "--visualization",
                        action='store_true', help="flag to enable visualization")
    parser.add_argument("--fullRes",help='save output of the network rescaled to full resolution',action="store_true")
    parser.add_argument("--max_disp",help="maximum value of disparity that can be predicted, clip value above",default=500,type=int)
    args = parser.parse_args()

    use_dir = False
    for f in [args.left, args.right]:
        if not os.path.exists(f):
            raise Exception('Unable to find: {}'.format(f))
        if os.path.isdir(f):
            use_dir = True

    # create output folders
    os.makedirs(args.output, exist_ok=True)

    # load inputs
    if use_dir:
        left_files = [os.path.join(args.left, f) for f in os.listdir(
            args.left) if f.endswith('.png') or f.endswith('.jpg')]
        right_files = [os.path.join(args.right, f) for f in os.listdir(
            args.right) if f.endswith('.png') or f.endswith('.jpg')]
    else:
        with open(args.left) as f_in:
            left_files = [x.strip() for x in f_in.readlines()]
        with open(args.right) as f_in:
            right_files = [x.strip() for x in f_in.readlines()]
        args.left = os.path.abspath(os.path.join(args.left, os.pardir))

    assert(len(left_files) == len(right_files))
    couples = [(l, r) for l, r in zip(left_files, right_files)]
    filename_queue = tf.train.input_producer(
        couples, element_shape=[2], num_epochs=1, shuffle=False)
    filenames = filename_queue.dequeue()
    left_fn, right_fn = filenames[0], filenames[1]
    left_raw = tf.read_file(left_fn)
    right_raw = tf.read_file(right_fn)

    left_img = tf.image.decode_image(left_raw)
    left_img.set_shape([None, None, 3])
    original_resolution = tf.shape(left_img)
    left_img = tf.image.convert_image_dtype(left_img, tf.float32)
    left_img = left_img - (100.0 / 255)
    left_img = pad_image(left_img)

    right_img = tf.image.decode_image(right_raw)
    right_img = tf.image.convert_image_dtype(right_img, tf.float32)
    right_img.set_shape([None, None, 3])
    right_img = right_img - (100.0 / 255)
    right_img = pad_image(right_img)

    target_shape = tf.placeholder(dtype=tf.int32, shape=[None])
    left_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,None,3])
    right_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,None,3])

    left_input = tf.expand_dims(left_placeholder,axis=0)
    right_input = tf.expand_dims(right_placeholder,axis=0)

    #left_img = tf.placeholder(dtype=tf.float32,shape=[1,Npne,None,3])
    #right_img  = tf.placeholder(dtype=tf.float32,)

    # build input batch
    #left_img_batch, right_img_batch, name_batch, resolution_batch = tf.train.batch([left_img, right_img, left_fn, original_resolution], args.batch_size, num_threads=4, capacity=args.batch_size * 100, allow_smaller_final_batch=True)

    # build model
    is_corr = args.corr_type != 'none'
    dispnet = DispNet(mode="inference", ckpt_path=args.checkpoint_path, batch_size=1, is_corr=is_corr, corr_type=args.corr_type, image_ops=[left_input, right_input])
    raw_prediction = dispnet.predictions_test[0]
    rescaled_prediction = tf.image.resize_images(raw_prediction,tf.shape(left_placeholder)[0:2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    cropped_prediction = tf.image.resize_image_with_crop_or_pad(rescaled_prediction,target_shape[0],target_shape[1])

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(dispnet.init)
        print("initialized")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("queue runners started")

        var_to_restore = get_var_to_restore_list(args.checkpoint_path, [], prefix="")
       
        print('Found {} variables to restore'.format(len(var_to_restore)))
        restorer = tf.train.Saver(var_list=var_to_restore)
        restorer.restore(sess, args.checkpoint_path)
        print('Weights restored')

        try:
            cc = 0
            saved = []
            while True:
                lefty,righty,f,ressy = sess.run([left_img,right_img,left_fn,original_resolution])
                raw_prediction_np, full_res_prediction_np = sess.run([raw_prediction,cropped_prediction],feed_dict={left_placeholder:lefty,right_placeholder:righty,target_shape:ressy})

                dest = f.decode('utf-8').replace(args.left, args.output)
                dest_folder = os.path.abspath(os.path.join(dest, os.pardir))
                os.makedirs(dest_folder, exist_ok=True)
                disparity = full_res_prediction_np if args.fullRes else raw_prediction_np
                immy = -1 * np.squeeze(disparity)
                immy = immy.astype(np.uint16)
                immy[np.where(immy>args.max_disp)]=args.max_disp
                cv2.imwrite(dest, immy)
                saved.append(dest)

                if args.visualization:
                    plt.figure('input_L')
                    plt.imshow(np.squeeze(lefty + (100 / 255)))

                    plt.figure('input_R')
                    plt.imshow(np.squeeze(righty + (100 / 255)))

                    plt.figure('prediction')
                    plt.imshow(np.squeeze(disparity))
                    plt.colorbar()
                    plt.show()
                cc += 1
                print('{}/{}'.format(cc,
                                     len(left_files)), end='\r')

        except tf.errors.OutOfRangeError:
            print('Done')

        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()

            saved = sorted(saved)
            with open(os.path.join(args.output,'prediction_list.txt'),'w+') as f_out:
                f_out.write('\n'.join(saved))
