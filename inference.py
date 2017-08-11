import argparse
import tensorflow as tf
import numpy as np
import os
from dispnet import DispNet
from util import get_var_to_restore_list
from matplotlib import pyplot as plt
import scipy.misc

INPUT_SIZE = (384, 768, 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=str,metavar="FILE", help='path to the folder with left image')
    parser.add_argument("--right", required=True,type=str,metavar='FILE',help="path to the folder with right image")
    parser.add_argument("-c", "--ckpt", dest="checkpoint_path", default=".", help='model checkpoint path')
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=4, type=int,help='batch size')
    parser.add_argument("--corr_type", dest="corr_type", type=str, default="tf",help="correlation layer realization",choices=['tf','cuda','none'])
    parser.add_argument("-o","--output",required=True,help="path were the predictions will be saved")
    parser.add_argument("-v","--visualization",action='store_true',help="flag to enable visualization")
    args = parser.parse_args()

    for f in [args.left,args.right]:
        if not os.path.exists(f):
            raise Exception('Unable to find directory: {}'.format(f))
    
    #create output folders
    os.makedirs(args.output,exist_ok=True)

    #load inputs
    left_files = [os.path.join(args.left,f) for f in os.listdir(args.left) if f.endswith('.png') or f.endswith('.jpg')]
    right_files = [os.path.join(args.right,f) for f in os.listdir(args.right) if f.endswith('.png') or f.endswith('.jpg')]
    assert(len(left_files)==len(right_files))
    couples = [(l,r) for l,r in zip(left_files,right_files)]
    filename_queue = tf.train.input_producer(couples,element_shape=[2],num_epochs=1,shuffle=False)
    filenames = filename_queue.dequeue()
    left_fn,right_fn = filenames[0],filenames[1]
    left_raw = tf.read_file(left_fn)
    right_raw = tf.read_file(right_fn)

    left_img = tf.image.decode_image(left_raw)
    left_img.set_shape([None,None,3])
    left_img = tf.image.convert_image_dtype(left_img, tf.float32)
    left_img = tf.image.resize_images(left_img,INPUT_SIZE[0:2])
    left_img = left_img-(100/255)

    right_img = tf.image.decode_image(right_raw)
    right_img = tf.image.convert_image_dtype(right_img,tf.float32)
    right_img.set_shape([None,None,3])
    right_img = tf.image.resize_images(right_img, INPUT_SIZE[0:2])
    right_img = right_img-(100/255)

    #build input batch
    left_img_batch,right_img_batch,name_batch = tf.train.batch([left_img,right_img,left_fn],args.batch_size,num_threads=4,capacity=args.batch_size*100)

    #build model
    is_corr = args.corr_type != 'none'
    dispnet = DispNet(mode="inference", ckpt_path=args.checkpoint_path, batch_size=args.batch_size,is_corr=is_corr, corr_type=args.corr_type,image_ops=[left_img_batch,right_img_batch])

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(dispnet.init)
        print("initialized")
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("queue runners started")

        var_to_restore = get_var_to_restore_list(args.checkpoint_path, [], prefix="model/")
        #trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #print(len(trainable))
        print('Found {} variables to restore'.format(len(var_to_restore)))
        restorer = tf.train.Saver(var_list=var_to_restore)
        restorer.restore(sess, args.checkpoint_path)
        print('Weights restored')

        try: 
            cc=0
            while True:  
                lefty, righty, files,  disparitys = sess.run([left_img_batch, right_img_batch, name_batch, dispnet.predictions_test[0]])
                
                for i,f in enumerate(files):
                    dest = f.decode('utf-8').replace(args.left,args.output)
                    immy= -1*np.squeeze(disparitys[i])
                    immy=immy.astype(np.uint8)
                    scipy.misc.imsave(dest,immy)

                    if args.visualization:
                        plt.figure('input_L')
                        plt.imshow(np.squeeze(lefty[i]+(100/255)))

                        plt.figure('input_R')
                        plt.imshow(np.squeeze(righty[i]+(100/255)))

                        plt.figure('prediction')
                        plt.imshow(np.squeeze(disparitys[i]))
                        plt.colorbar()
                        plt.show()
                cc+=1
                print('{}/{}'.format(cc*args.batch_size,len(left_files)),end='\r')
        
        except tf.errors.OutOfRangeError:
            print('Done')

        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()
