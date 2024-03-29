import os
import shutil
from tqdm import tqdm
from utils.util import get_images

if __name__ == '__main__':
    root = r'D:\llf\dataset\danyang\2024_dataset\20240327'
    images_dir = os.path.join(root, 'imgs')
    output_dir = os.path.join(root, 'good-g')
    name = 'B4'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert os.path.exists(output_dir) is True

    for image in tqdm(get_images(images_dir)):
        basename = os.path.basename(image)
        if name in basename:
            new_path = os.path.join(output_dir, basename)
            shutil.move(image, new_path)
