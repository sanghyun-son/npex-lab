from os import path
import random
import argparse

import utils
from data import backbone
from model import simple

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
cfg = parser.parse_args()
seed = 20190715


def main():
    # Random seed initialization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define your dataloader here
    loader_train = DataLoader(
        backbone.RestorationData(
            '../DIV2K_sub/train/target',
            '../DIV2K_sub/train/target',
            training=True,
        ),
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    loader_eval = DataLoader(
        backbone.RestorationData(
            '../DIV2K_sub/eval/target',
            '../DIV2K_sub/eval/target',
            training=False,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    writer = tensorboard.SummaryWriter(
        log_dir=path.join('..', 'experiment', cfg.save)
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device('cpu')

    # Make a CNN
    net = simple.Simple()
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
        print('Epoch {}'.format(epoch))
        total_iteration = 0
        net.train()
        for batch, (x, t) in enumerate(tqdm.tqdm(loader_train)):
            if (batch + 1) % 100 or batch == 0:
                writer.add_images('training_input', x, global_step=total_iteration)
                writer.add_images('training_target', t, global_step=total_iteration)
            x = x.to(device)
            t = t.to(device)

            optimizer.zero_grad()
            y = net(x)
            loss = F.mse_loss(y, t)
            loss.backward()
            optimizer.step()

            total_iteration += 1
            writer.add_scalar('training_loss', loss.item(), global_step=total_iteration)


    def do_eval(epoch: int):
        net.eval()
        for x, t in tqdm.tqdm(loader_eval):
            x = x.to(device)
            t = t.to(device)
            # Define your evaluation loop here

    # Outer loop
    for i in tqdm.trange(cfg.epochs):
        do_train(i + 1)
        do_eval(i + 1)


if __name__ == '__main__':
    main()
