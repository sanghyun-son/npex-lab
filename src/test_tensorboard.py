from os import path
import shutil

from data import backbone
from data import preprocessing as pp

import torch
from torch.utils import tensorboard

import imageio


def quantize(x):
    x = (x + 1) * 127.5
    x = x.round()
    x = x.byte()
    return x

def main():
    log_dir = path.join('sample', 'log')
    shutil.rmtree(log_dir, ignore_errors=True)
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    img_input = imageio.imread(path.join('sample', 'butterfly_lr.png'))
    img_target = imageio.imread(path.join('sample', 'butterfly.png'))

    img_input, img_target = pp.to_tensor(img_input, img_target)
    writer.add_image('img_input', quantize(img_input))
    writer.add_image('img_target', quantize(img_target))

    dir_input = '[your_path]'
    dir_target = '[your_path]'
    data_test = backbone.RestorationData(dir_input, dir_target, method='direct')
    x, y = data_test[0]
    writer.add_image('patch_input', quantize(x)) 
    writer.add_image('patch_target', quantize(y)) 
    # Bug?
    writer.add_image('patch_target', quantize(y)) 


if __name__ == '__main__':
    main()

