# Imports
import os
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
import numpy as np

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

class SeaIce(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.data_files = []
        for dir in root_dir:
            classes = os.listdir(dir)
            for c in classes:
                for f in os.walk(os.path.join(dir , c)):
                     self.data_files = self.data_files + [(os.path.join(f[0],path),c) for path in f[2]]
                     print(self.data_files)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img_path = self.data_files[index][0]
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.data_files[index][1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


class sea_ice_scale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self) :
        # assert isinstance(output_size, (int, tuple))
        # self.output_size = output_size
        pass

    def __call__(self, sample):
        # print('sdasd')
        u = np.zeros(sample.shape, 'float32')
        u[:, :, 0:2] = sample[:, :, 0:2] / 255
        u[:,:,2] = sample[:, :, 2] / 46
        # u = np.rollaxis(u,2,0)
        return u