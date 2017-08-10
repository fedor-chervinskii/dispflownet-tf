import re
import os
import sys
import glob
import logging
import datetime
import numpy as np

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dims = file.readline()
    try:
        width,height = list(map(int,dims.split()))
    except:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width,1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

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
    ft3d_samples_filenames['PFM']=True
    return ft3d_samples_filenames

def trainingLists_conf(filename_train, filename_test):
    for f in [filename_train,filename_test]:
        if not os.path.exists(f):
            raise Exception('File not found: {}'.format(f)) 
    dataset={}
    for label,filename in zip(['TRAIN','TEST'],[filename_train,filename_test]):
        with open(filename) as f:
            lines=[l.strip() for l in f.readlines()]
        dataset[label]=[l.split(';') for l in lines] 

    dataset['PFM']=(dataset['TRAIN'][0][2].split('.')[-1]=='pfm')
    return dataset

def init_logger(log_path, name="dispnet"):
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)
    logfile = os.path.join(log_path, "%s-%s.log" % (name, datetime.datetime.today()))
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setLevel(logging.INFO)
    root.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.terminator = ""
    root.addHandler(consoleHandler)
    logging.debug("Logging to %s" % logfile)
