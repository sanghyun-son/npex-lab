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

        self.conv4 = nn.Conv2d(64, 256, 3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(64, 3, 3, padding=1)

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
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.ps(x)
        x = self.relu4(x)

        x = self.conv5(x)

        return x


if __name__ == '__main__':
    net = RestorationNet()
    x = torch.randn(1, 3, 64, 64)
    print(x.size())
    print(net(x).size())

