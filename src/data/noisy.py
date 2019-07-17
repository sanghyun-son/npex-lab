from data import backbone
import torch

class NoisyData(backbone.RestorationData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, x, y):
        x, y = super().preprocess(x, y) # x, y are torch.Tensor
        if self.training:
            n = (40 / 255) * torch.randn_like(x)
            x = x + n

        return x, y

