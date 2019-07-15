import torch
from torch import nn
from torch.nn import functional as F


def ResBlock(nn.Module):
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
        super(self).__init__()
        # Finish the implementation

    def forward(self, x):
        # Finish the implementation

        return x

