from os import path

import math
import torch


def quantize(x):
    x = (x + 1) * 127.5
    x = x.clamp(min=0, max=255)
    x = x.round()
    x /= 255

    return x

def psnr(
        x: torch.Tensor,
        y: torch.Tensor,
        luminance: bool = False,
        crop: int = 4) -> float:

    '''
    Calculate a PSNR value between x and y.

    Args:
        x (Tensor): An image to calculate the PSNR valuei [-1, 1].
        y (Tensor): A reference image [-1, 1].
        luminance (bool, optional): If set to True,
            calculate the PSNR on a luminance channel only.
        crop (int, optional): Crop n pixels from image boundaries.

    Return:
        (float): A PSNR value between x and y.
    '''

    if luminance:
        pass

    diff = x - y    # B x C x H x W
    diff = diff[..., crop:-crop, crop:-crop]
    mse = diff.pow(2).mean().item()
    max_square = 4
    psnr = 10 * math.log10(max_square / mse)

    return psnr

# !!Do not modify below lines!!
def dir_path(p: str) -> str:
    if path.isdir(p):
        return p
    else:
        raise NotADirectoryError(p)


if __name__ == '__main__':
    import imageio
    import data.preprocessing as pp
    img_noise = imageio.imread('sample/butterfly_noise.png')
    img_clean = imageio.imread('sample/butterfly.png')

    img_noise, img_clean = pp.to_tensor(img_noise, img_clean)

    print(psnr(img_noise, img_clean))

