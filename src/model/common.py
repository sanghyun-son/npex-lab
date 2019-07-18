import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    '''
    A basic implementation of the residual block

    Args:
        n_feats (int): The number of internal features.
        activation (str, optional): Choice of the activation function.

    Structure:
          |
        --|
        |Conv (n_feats x n_feats)
        |ReLU
        |Conv (n_feats x n_feats)
        --+
          |
    '''

    def __init__(self, n_feats, activation='relu'):
        super().__init__()
        # Finish the implementation
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)

    def forward(self, x):
        # Finish the implementation
        r = self.conv1(x)
        r = self.relu(r)
        r = self.conv2(r)

        y = x + r

        return y

