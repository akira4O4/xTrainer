import os
import shutil
from tqdm import tqdm
from utils.util import get_images

if __name__ == '__main__':
    images_dir = r'D:\data\0danyang\20240124\B\good'
    output_dir = r'D:\data\0danyang\20240124\good-f'
    name = 'B5'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert os.path.exists(output_dir) is True

    for image in tqdm(get_images(images_dir)):
        basename = os.path.basename(image)
        if name in basename:
            new_path = os.path.join(output_dir, basename)
            shutil.move(image, new_path)
