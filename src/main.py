import os
from os import path
import random
import argparse
import importlib

import utils
from data import backbone
from data import noisy
from model import discriminator

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import utils as vutils

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
parser.add_argument('-p', '--pretrained', type=str)
parser.add_argument('-g', '--gan', action='store_true')
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
        # In case of deblurring/super-resolution
        # backbone.RestorationData(
        # In case of denoising
        # noisy.NoisyData
        backbone.RestorationData(
            '../DIV2K_sub/train/input_x4',
            '../DIV2K_sub/train/target',
            training=True,
            p=32,
        ),
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    loader_eval = DataLoader(
        # In case of deblurring/super-resolution
        # backbone.RestorationData(
        # In case of denoising
        # noisy.NoisyData
        backbone.RestorationData(
            '../DIV2K_sub/eval/input_x4',
            '../DIV2K_sub/eval/target',
            training=False,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=8,
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
    print(net)

    # Fine-tuning implementation for scale transfer learning (EDSR)
    keys_to_remove = []
    if cfg.pretrained:
        pt = torch.load(cfg.pretrained)['model']
        for k in pt.keys():
            # We don't want weights from the upsampling module
            # if we want to initialize 4x model from 2x
            '''
            if 'us' in k:
                # Save keys to remove
                keys_to_remove.append(k)
            '''
            pass
        # Remove unwanted keys from state_dict
        for k in keys_to_remove:
            pt.pop(k)

        net.load_state_dict(pt, strict=False)

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

    if cfg.gan:
        dis = discriminator.Discriminator()
        dis = dis.to(device)
        dis_params = [p for p in dis.parameters() if p.requires_grad]
        dis_optimizer = optim.Adam(dis_params, lr=1e-4)
        dis_scheduler = lr_scheduler.MultiStepLR(
            dis_optimizer,
            milestones=[int(0.5 * cfg.epochs), int(0.75 * cfg.epochs)],
            gamma=0.5,
        )

    def do_train(epoch: int):
        global total_iteration
        print('Epoch {}'.format(epoch))
        net.train()
        tq = tqdm.tqdm(loader_train)
        for batch, (x, t) in enumerate(tq):
            # x: LR input   B x C x H x W
            # t: HR target  B x C x sH x sW
            x = x.to(device)
            t = t.to(device)

            optimizer.zero_grad()
            # y: SR output  B x C x sH x sW
            y = net(x)
            # Replaced to L1 loss (EDSR, LapSRN)
            loss = F.l1_loss(y, t)    

            # We will implement GAN
            if cfg.gan:
                # Reset the discrimator gradient
                dis_optimizer.zero_grad()
                #dis_loss_naive = -((dis(t).sigmoid()).log() + (1 - dis(y).sigmoid()).log()).mean()
                dis_real = dis(t)
                dis_fake = dis(y.detach())  # For technical issue
                target_real = torch.ones_like(dis_real)
                target_fake = torch.zeros_like(dis_fake)
                dis_loss_real = F.binary_cross_entropy_with_logits(
                    dis_real, target_real
                )
                dis_loss_fake = F.binary_cross_entropy_with_logits(
                    dis_fake, target_fake
                )
                dis_loss = dis_loss_real + dis_loss_fake
                
                # Backpropagation
                dis_loss.backward()
                dis_optimizer.step()
            
                #gen_loss_naive = -(dis(y).sigmoid()).log().mean()
                dis_gen = dis(y)
                target_gen = torch.ones_like(dis_gen)
                gen_loss = F.binary_cross_entropy_with_logits(
                    dis_gen, target_gen
                )
                loss = loss + 0.01 * gen_loss

            tq.set_description('{:.4f}'.format(loss.item()))
            loss.backward()
            optimizer.step()

            total_iteration += 1
            # Tensorboard batch logging
            if total_iteration % 500 == 0 and epoch <= 5:
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

            if total_iteration % 10 == 0:
                writer.add_scalar(
                    'training_loss', loss.item(), global_step=total_iteration
                )
                if cfg.gan:
                    writer.add_scalar(
                        'training_dis_real',
                        dis_loss_real.item(),
                        global_step=total_iteration,
                    )
                    writer.add_scalar(
                        'training_dis_fake',
                        dis_loss_fake.item(),
                        global_step=total_iteration,
                    )
                    writer.add_scalar(
                        'training_dis_loss',
                        dis_loss.item(),
                        global_step=total_iteration,
                    )
                    writer.add_scalar(
                        'training_gen',
                        gen_loss.item(),
                        global_step=total_iteration,
                    )

    def do_eval(epoch: int):
        net.eval()
        avg_loss = 0
        avg_psnr = 0
        with torch.no_grad():
            for idx, (x, t) in enumerate(tqdm.tqdm(loader_eval)):
                x = x.to(device)
                t = t.to(device)

                y = net(x)
                # Replaced to L1 loss (EDSR, LapSRN)
                avg_loss += F.l1_loss(y, t)
                avg_psnr += utils.psnr(y, t)

                if epoch < 5 or epoch % 10 == 0:
                    # Code for saving image
                    # 1 x C x H x W
                    # y \in [-1, 1]
                    y_save = (y + 1) * 127.5    # [0, 255]
                    y_save = y_save.clamp(min=0, max=255)
                    y_save = y_save.round()
                    y_save = y_save / 255
                    output_dir = path.join(log_dir, 'output')
                    os.makedirs(output_dir, exist_ok=True)
                    vutils.save_image(
                        y_save, path.join(output_dir, '{:0>2}.png'.format(idx + 1))
                    )

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
            to_save = {}
            to_save['model'] = net.state_dict()
            to_save['optimizer'] = optimizer.state_dict()
            to_save['misc'] = avg_psnr
            torch.save(to_save, path.join(log_dir, 'checkpoint_{:0>2}.pt'.format(epoch)))

        print('PSNR {:.2f}dB'.format(avg_psnr))

    # Outer loop
    for i in range(cfg.epochs):
        do_train(i + 1)
        do_eval(i + 1)
        # Learning rate adjustment
        scheduler.step()


if __name__ == '__main__':
    main()

