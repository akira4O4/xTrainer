import os
import cv2
from utils.util import get_images
from tqdm import tqdm
import shutil

if __name__ == '__main__':

    a_t = 500
    root = r'D:\llf\dataset\danyang\2024_dataset\16\good-f'
    output = r'D:\llf\dataset\danyang\2024_dataset\16\output'
    if not os.path.exists(output):
        os.makedirs(output)

    for image in tqdm(get_images(root)):
        basename = os.path.basename(image)
        name, ext = os.path.splitext(basename)
        area = int(name.split('a')[-1])
        if area >= a_t:
            shutil.copy(image, os.path.join(output, basename))
