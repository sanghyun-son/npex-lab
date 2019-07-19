import torch
from torch import nn


class Discriminator(nn.Module):
    '''

    Note:
        From Ledig et al.,
        "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
        See https://arxiv.org/pdf/1609.04802.pdf for more detail.
    '''

    def __init__(self):
        super().__init__()
        depth = 8
        in_channels = 3
        out_channels = 64
        stride = 1
        m = []

        for i in range(depth):
            # Conv-BN-LeakyReLU
            m.append(nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ))
            m.append(nn.BatchNorm2d(out_channels))
            m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            in_channels = out_channels

            # Reduce resolution at 2, 4, 6, 8-th Conv
            stride = 2 - (i % 2)
            if i == depth - 1:
                break
            elif i % 2 == 1:
                out_channels *= 2

        # Feature extraction module
        self.features = nn.Sequential(*m)
        '''
        PatchGAN style
        From Isola et al.,
        "Image-to-Image Translation with Conditional Adversarial Networks"
        (pix2pix)
        See https://arxiv.org/pdf/1611.07004.pdf for more detail.
        '''
        # Fully-convolutional classifier
        self.cls = nn.Conv2d(out_channels, 1, 3, padding=1, bias=False)

        # Discriminator initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # See DCGAN
                m.weight.data.normal_(0.0, 0.02)
                '''
                Apply spectral normalization to all convolutions

                Note:
                    From Miyato et al.,
                    "Spectral Normalization for Generative Adversarial Networks"
                    See https://arxiv.org/pdf/1802.05957.pdf for more detail.
                '''
                nn.utils.spectral_norm(m, name='weight', n_power_iterations=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Note:
            The output is unbounded.
            Use a sigmoid function or nn.bce_with_logits to avoid
            unexpected behaviors.
        '''
        x = self.features(x)
        x = self.cls(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 128, 128)
    d = Discriminator()
    print(d)

    print(x.size())
    print(d(x).size())

