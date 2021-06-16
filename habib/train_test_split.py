from PIL import Image
from os import listdir
from os.path import isfile, join
import random
#import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

patchsize = '32'
dir = '/home/skh018/PycharmProjects/EE-D2.7/dataset/HH_Filtered_HV_IA_db_scaledBy_40_160_patchSize_32//new_patches/'
out = '/home/skh018/PycharmProjects/EE-D2.7/dataset/HH_Filtered_HV_IA_db_scaledBy_40_160_patchSize_32//new_patches//80-20/'
path_0 = [dir + '/0/', dir  + '/1/']
path_1 = [ dir  + '/2/', dir  + '/3/', dir  + '/4/', dir  + '/5/', dir + '/6/']
classes = ['0.0','1.0']
train_dir = out + 'train+val/'
test_dir =  out + 'test/'
unlabed_dir = train_dir + 'unlabeled_data'

for path in classes:
    if not os.path.exists(train_dir + path):
        os.makedirs(train_dir + path)
        # Path(train_dir + path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(test_dir + path):
        os.makedirs(test_dir + path)
        # Path(test_dir + path).mkdir(parents=True, exist_ok=True)

if not os.path.exists(unlabed_dir):
    os.makedirs(unlabed_dir)


image_paths = []
for path in path_0:
    for root, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            image_path = os.path.join(root, fname)
            image_paths.append(image_path)
sea_index = image_paths.__len__()

for path in path_1:
    for root, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            image_path = os.path.join(root, fname)
            image_paths.append(image_path)

ice_index = image_paths.__len__() - sea_index

y = np.ones(image_paths.__len__())
y[0:sea_index-1] = 0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_paths, y, test_size=0.20, random_state=33)

x_validation, x_unlabeled, y_validation, y_unlabeled = train_test_split(X_test, y_test, test_size=0.01, random_state=33)


for i in range(0,len(y_train)):
    copy_cmd = "cp {} {}train+val/{}".format(X_train[i],out,y_train[i])
    os.system(copy_cmd)
for i in range(0,len(y_validation)):
    copy_cmd = "cp {} {}test/{}".format(x_validation[i],out,y_validation[i])
    os.system(copy_cmd)

for i in range(0,len(y_unlabeled)):
    copy_cmd = "cp {} {}train+val/unlabeled_data/".format(x_unlabeled[i],out)
    os.system(copy_cmd)

