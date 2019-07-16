import glob
import numpy as np
import imageio

img_list = sorted(glob.glob('../DIV2K_sub/eval/target/*.png'))

for img_name in img_list:
    # img: np.uint8
    img = imageio.imread(img_name).astype(np.float)
    n = 20 * np.random.randn(*img.shape)    # np.float
    # Danger
    img_noise = img + n
    img_noise = img_noise.clip(min=0, max=255)
    img_noise = img_noise.round()
    img_noise = img_noise.astype(np.uint8)

    imageio.imwrite(img_name.replace('target', 'input'), img_noise)


