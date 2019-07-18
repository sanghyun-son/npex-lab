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

        n_feats = 64
        self.conv1 = nn.Conv2d(3, n_feats, 3, padding=1)

        m = []
        for _ in range(16):
            m.append(common.ResBlock(n_feats))

        m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.resblocks = nn.Sequential(*m)


        self.us_conv1 = nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1)
        self.us_ps1 = nn.PixelShuffle(2)

        self.us_conv2 = nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1)
        self.us_ps2 = nn.PixelShuffle(2)

        self.conv2 = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        SimpleNet forward function.

        Args:
            x (B x 3 x H x W Tensor): An input image

        Return:
            (B x 3 x 4H x 4W Tensor): An output image
        '''
        # Autograd will keep the history.
        x = self.conv1(x)

        # Calculate global residual
        r = self.resblocks(x)
        y = x + r

        y = self.us_conv1(y)
        y = self.us_ps1(y)
        y = self.us_conv2(y)
        y = self.us_ps2(y)
        y = self.conv2(y)

        return y


if __name__ == '__main__':
    net = RestorationNet()
    x = torch.randn(1, 3, 64, 64)
    print(x.size())
    print(net(x).size())

