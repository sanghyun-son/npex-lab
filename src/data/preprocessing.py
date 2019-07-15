import random
import torch

import numpy as np
from skimage import color

def crop(x, y, p=32, training=True):
    '''
    Crop image patches from input and target.

    Args:
        x (np.array): H x W (x C) array.
        y (np.array): sH x sW (x C) array.
        p (int, optional): Patch size.
        training (bool, optional): Don't crop patches during the evaluation.

    Return:
        x (np.array): p x p (x C) array.
        y (np.array): sp x sp (x C) array.

    '''
    if not training:
        return x, y

    h = x.shape[0]
    w = x.shape[1]

    py = random.randrange(0, h - p + 1)
    px = random.randrange(0, w - p + 1)

    x = x[py:(py + p), px:(px + p)]
    y = y[py:(py + p), px:(px + p)]

    return x, y

def set_channel(x, y, c=3):
    '''
    Convert images to have a specific number of channels.
    
    Args:
        x (np.array): H x W (x C) array.
        y (np.array): sH x sW (x C) array.
        c (int, optional): The number of color channels to be set.

    Return:
        x (np.array): H x W x c array.
        y (np.array): sH x sW x c array.

    '''
    # Finish the implementation

    return x, y

def augment(x, y, hflip=True, vflip=True, rot=True):
    '''
    Apply random transforms to the training patches.

    Args:
        x (np.array): H x W x C array.
        y (np.array): sH x sW x C array.
        hflip (bool, optional): Randomly apply horizontal flip.
        vflip (bool, optional): Randomly apply vertical flip.
        rot (bool, optional): Randomly apply 90-degree rotation.

    Return:
        x (np.array): Transformed H x W x C array.
        y (np.array): Transformed sH x sW x C array.

    '''
    # Advanced: Finish the implementation

    return x, y

def to_tensor(x, y):
    '''
    Convert numpy arrays to PyTorch Tensors

    Args:
        x (np.array): H x W x C array.
        y (np.array): sH x sW x C array.

    Return:
        x (torch.Tensor): C x H x W Tensor in [-1, 1].
        y (torch.Tensor): C x sH x sW Tensor in [-1, 1].

    '''
    # Finish the implementation
    x = np.transpose(x, (2, 0, 1))
    # C x H x W / uint8
    x = x.astype(np.float)
    # For efficient memory allocation...
    x = np.ascontiguousarray(x)
    # Now we have torch.FloatTensor [0, 255]
    x = torch.from_numpy(x)
    x /= 127.5      # [0, 2]
    x -= 1          # [-1, 1]

    y = np.transpose(y, (2, 0, 1))
    y = y.astype(np.float)
    y = np.ascontiguousarray(y)
    y = torch.from_numpy(y)
    y /= 127.5      # [0, 2]
    y -= 1          # [-1, 1]

    return x, y

