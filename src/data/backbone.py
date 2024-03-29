from os import path
import glob
import random
import pickle

import data.preprocessing as pp

import torch
from torch.utils import data

import imageio

class RestorationData(data.Dataset):
    '''
    A backbone dataset class for the deep image restoration.

    Args:
        dir_input (str): A directory for input images.
        dir_target (str): A directory for target images.
        p (int, optional): Patch size.
        c (int, optional): The number of color channels.
        training (bool, optional): Set False to indicate evaluation dataset.
        method (str, optional): Choice ('direct', 'predecode', 'preload')

    '''

    def __init__(
            self, dir_input, dir_target,
            p=64, c=3, training=True, method='predecode'):

        self.dir_input = dir_input
        self.dir_target = dir_target
        self.p = p
        self.c = c
        self.training = training
        self.method = method

        self.img_input = sorted(glob.glob(path.join(dir_input, '*.png')))
        self.img_target = sorted(glob.glob(path.join(dir_target, '*.png')))
        if len(self.img_input) != len(self.img_target):
            raise IndexError('both lists should have the same lengths.')

        if method == 'predecode':
            # Finish the implementation with pickle
            pass
        elif method == 'preload':
            # Implement it if you want
            pass

    def __getitem__(self, idx):
        '''
        Get an idx-th input-target pair.

        Args:
            idx (int): An index of the pair.

        Return:
            (C x H x W Tensor): An input image.
            (C x sH x sW Tensor): A target image.
        '''
        idx %= len(self.img_input)

        if self.method == 'direct':
            # Will load images on-the-fly
            x = imageio.imread(self.img_input[idx])
            y = imageio.imread(self.img_target[idx])
        elif self.method == 'predecode':
            # Implement it if you want
            pass
        elif self.method == 'preload':
            # Finish the implementation with pickle
            pass

        x, y = self.preprocess(x, y)

        return x, y

    def preprocess(self, x, y):
        #x, y = pp.crop(x, y, p=self.p, training=self.training)
        #x, y = pp.set_channel(x, y, c=self.c)
        # If you've implemented the augmentation
        #x, y = pp.augment(x, y)
        #x, y = pp.to_tensor(x, y)

        return x, y

    def __len__(self):
        '''
        Get the length of the dataset.
        
        Return:
            (int): Total number of the input-target pairs in the dataset.
        '''
        if self.training:
            return 3200
        else:
            return len(self.img_input)

