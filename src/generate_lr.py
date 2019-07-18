import os
import glob
from PIL import Image
import tqdm

scales = [2, 4]
for split in ('train', 'eval'):
    img_list = sorted(glob.glob('../DIV2K_sub/{}/target/*.png'.format(split)))
    for img_name in tqdm.tqdm(img_list):
        # img: np.uint8
        hr = Image.open(img_name)
        w, h = hr.size
        for s in scales:
            os.makedirs('../DIV2K_sub/{}/input_x{}'.format(split, s), exist_ok=True)
            lw = w // s
            lh = h // s

            lr = hr.resize((lw, lh), resample=Image.BICUBIC)
            lr.save(img_name.replace('target', 'input_x{}'.format(s)))


