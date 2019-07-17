import torch
from torch import nn

class RestorationNet(nn.Module):
    '''
    A simple image2image CNN.
    '''

    def __init__(self):
        # This line is very important!
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        SimpleNet forward function.

        Args:
            x (B x 3 x H x W Tensor): An input image

        Return:
            (B x 3 x H x W Tensor): An output image
        '''
        # Autograd will keep the history.
        r = self.conv1(x)
        r = self.relu1(r)

        r = self.conv2(r)
        r = self.relu2(r)

        r = self.conv3(r)
        r = self.relu3(r)

        r = self.conv4(r)
        y = x + r
        return y

