from os import path
import random
import argparse
import importlib

import utils
from data import backbone
from data import noisy

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils import tensorboard
from torch.utils.data import DataLoader

import numpy as np
import tqdm

# Argument parsing
parser = argparse.ArgumentParser('NPEX Image Restoration Lab')
parser.add_argument('-i', '--input', type=utils.dir_path)
parser.add_argument('-t', '--target', type=utils.dir_path)
parser.add_argument('-e', '--epochs', type=int, default=20)
parser.add_argument('-s', '--save', type=str, default='test')
parser.add_argument('-u', '--sub_save', type=str)
parser.add_argument('-m', '--model', type=str, default='simple')
cfg = parser.parse_args()
seed = 20190715
total_iteration = 0


def main():
    # Random seed initialization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define your dataloader here
    loader_train = DataLoader(
        noisy.NoisyData(
            '../DIV2K_sub/train/target',
            '../DIV2K_sub/train/target',
            training=True,
            p=64,
        ),
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    loader_eval = DataLoader(
        noisy.NoisyData(
            '../DIV2K_sub/eval/input',
            '../DIV2K_sub/eval/target',
            training=False,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    log_dir = path.join('..', 'experiment', cfg.save)
    if cfg.sub_save:
        log_dir = path.join(log_dir, cfg.sub_save)

    writer = tensorboard.SummaryWriter(log_dir)

    # CUDA configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device('cpu')

    # Make a CNN
    net_module = importlib.import_module('.' + cfg.model, package='model')
    net = net_module.RestorationNet()
    net = net.to(device)

    # Will be supported later...
    '''
    writer.add_graph(
        net,
        input_to_model=torch.randn(1, 3, 64, 64).to(device),
    )
    '''

    # Set up an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4)

    # Set up a learning rate scheduler
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * cfg.epochs), int(0.75 * cfg.epochs)],
        gamma=0.5,
    )

    def do_train(epoch: int):
        global total_iteration
        print('Epoch {}'.format(epoch))
        net.train()
        tq = tqdm.tqdm(loader_train)
        for batch, (x, t) in enumerate(tq):
            x = x.to(device)
            t = t.to(device)

            optimizer.zero_grad()
            y = net(x)
            loss = F.mse_loss(y, t)
            tq.set_description('{:.4f}'.format(loss.item()))
            loss.backward()
            optimizer.step()

            total_iteration += 1
            # Tensorboard batch logging
            if total_iteration % 100 == 0:
                writer.add_images(
                    'training_input',
                    utils.quantize(x.cpu()),
                    global_step=total_iteration
                )
                writer.add_images(
                    'training_target',
                    utils.quantize(t.cpu()),
                    global_step=total_iteration
                )
                writer.add_images(
                    'training_output',
                    utils.quantize(y.cpu()),
                    global_step=total_iteration
                )

            writer.add_scalar('training_loss', loss.item(), global_step=total_iteration)


    def do_eval(epoch: int):
        net.eval()
        avg_loss = 0
        avg_psnr = 0
        with torch.no_grad():
            for x, t in tqdm.tqdm(loader_eval):
                x = x.to(device)
                t = t.to(device)

                y = net(x)
                avg_loss += F.mse_loss(y, t)
                avg_psnr += utils.psnr(y, t)

            avg_loss /= len(loader_eval)
            avg_psnr /= len(loader_eval)

            # Tensorboard logging for evaluation
            writer.add_scalar(
                'evaluation_loss',
                avg_loss.item(),
                global_step=epoch
            )
            writer.add_scalar(
                'evaluation_psnr',
                avg_psnr,
                global_step=epoch
            )

    # Outer loop
    for i in range(cfg.epochs):
        do_train(i + 1)
        do_eval(i + 1)
        # Learning rate adjustment
        scheduler.step()


if __name__ == '__main__':
    main()

