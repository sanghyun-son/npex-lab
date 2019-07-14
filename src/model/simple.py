import torch
from torch import nn

class Simple(nn.Module):

    '''
    A simple image2image CNN.
    '''
    def __init__(self):
        # This line is very important!
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        SimpleNet forward function.

        Args:
            x (B x 3 x H x W Tensor): An input image

        Return:
            (B x 3 x H x W Tensor): An output image
        '''
        # Autograd will keep the history.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x
