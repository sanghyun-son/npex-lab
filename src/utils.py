from os import path

import torch


def quantize(x):
    x = (x + 1) * 127.5
    x = x.round()
    x = x.byte()
    return x

def psnr(
        x: torch.Tensor,
        y: torch.Tensor,
        luminance: bool = False,
        crop: int = 4) -> float:

    '''
    Calculate a PSNR value between x and y.

    Args:
        x (Tensor): An image to calculate the PSNR value.
        y (Tensor): A reference image.
        luminance (bool, optional): If set to True,
            calculate the PSNR on a luminance channel only.
        crop (int, optional): Crop n pixels from image boundaries.

    Return:
        (float): A PSNR value between x and y.
    '''

    if luminance:
        pass
    else:
        pass

    return 0

# !!Do not modify below lines!!
def dir_path(p: str) -> str:
    if path.isdir(p):
        return p
    else:
        raise NotADirectoryError(p)
