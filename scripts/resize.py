import cv2
import os
from utils.util import get_images
from tqdm import tqdm

if __name__ == '__main__':
    root = r'D:\llf\dataset\danyang\0\gen_bad\20240406b-cls-temp-e-done\classification\cls\wenli.checked'

    wh = (576, 576)
    output = root + f'-w{wh[0]}h{wh[1]}'

    if not os.path.exists(output):
        os.makedirs(output)

    images = get_images(root)
    for image in tqdm(images):
        basename = os.path.basename(image)
        im = cv2.imread(image)
        im = cv2.resize(im, wh)
        cv2.imwrite(os.path.join(output, basename), im)
