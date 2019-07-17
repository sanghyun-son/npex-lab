import torch
from torch import nn
from torch.nn import init

class RestorationNet(nn.Module):
    '''
    A simple image2image CNN.
    '''

    def __init__(self):
        # This line is very important!
        super().__init__()
        m = [
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(8):
            m.append(nn.Conv2d(64, 64, 3, padding=1))
            m.append(nn.ReLU(inplace=True))

        m.append(nn.Conv2d(64, 3, 3, padding=1))
        self.seq = nn.Sequential(*m)

        for conv in self.modules():
            if isinstance(conv, nn.Conv2d):
                init.kaiming_normal_(conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        SimpleNet forward function.

        Args:
            x (B x 3 x H x W Tensor): An input image

        Return:
            (B x 3 x H x W Tensor): An output image
        '''
        # Autograd will keep the history.
        y = x + self.seq(x)
        return y

