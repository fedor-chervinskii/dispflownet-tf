import re
import os
import glob
import numpy as np

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

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
    return ft3d_samples_filenames