import torch
from torch import nn
from model import common

class RestorationNet(nn.Module):

    '''
    A simple image2image CNN.
    '''
    def __init__(self):
        # This line is very important!
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.res1 = common.ResBlock(32)
        self.res2 = common.ResBlock(32)
        self.res3 = common.ResBlock(32)
        self.res4 = common.ResBlock(32)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        self.us_conv = nn.Conv2d(32, 128, 3, padding=1)
        self.us_ps = nn.PixelShuffle(2)

        self.conv3 = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        SimpleNet forward function.

        Args:
            x (B x 3 x H x W Tensor): An input image

        Return:
            (B x 3 x 2H x 2W Tensor): An output image
        '''
        # Autograd will keep the history.
        x = self.conv1(x)

        r = self.res1(x)
        r = self.res2(r)
        r = self.res3(r)
        r = self.res4(r)
        r = self.conv2(r)

        y = x + r

        y = self.us_conv(y)
        y = self.us_ps(y)
        y = self.conv3(y)

        return y


if __name__ == '__main__':
    net = RestorationNet()
    x = torch.randn(1, 3, 64, 64)
    print(x.size())
    print(net(x).size())

