import os
import shutil
from tqdm import tqdm
from utils.util import get_images

if __name__ == '__main__':
    root = r'D:\llf\dataset\danyang\2024_dataset\20240306B\未命名文件夹\PNG-13-星期三good.tar\PNG-13-星期三good'
    images_dir = os.path.join(root, 'PNG-13-星期三')
    output_dir = os.path.join(root, 'good-f')
    name = 'B5'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert os.path.exists(output_dir) is True

    for image in tqdm(get_images(images_dir)):
        basename = os.path.basename(image)
        if name in basename:
            new_path = os.path.join(output_dir, basename)
            shutil.move(image, new_path)
