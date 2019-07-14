from os import path
import time

from data import backbone

import torch

import tqdm


def run_test(dataset_class):
    t_begin = time.time()
    for idx in tqdm.trange(len(dataset_class)):
        x, y = dataset_class[idx]

    t_end = time.time()
    print('Elapsed time: {:.1f}'.format(t_end - t_begin))

def main():
    dir_input = path.join('..', 'dataset', 'DIV2K', 'DIV2K_train_HR')
    dir_target = path.join(
        '..', 'dataset', 'DIV2K', 'DIV2K_train_LR_bicubic', 'X4'
    )

    data_direct = backbone.RestorationData(
        dir_input, dir_target, method='direct'
    )
    print('Direct method')
    run_test(data_direct)

    data_predecode = backbone.RestorationData(
        dir_input, dir_target, method='predecode'
    )
    print('Pre-decode method')
    run_test(data_predecode)

    # If you want to test.
    # Always monitor the memory usage!
    '''
    data_preload = backbone.RestorationData(
        dir_input, dir_target, method='preload'
    )
    print('Pre-load method')
    run_test(data_predecode)
    '''


if __name__ == '__main__':
    main()
