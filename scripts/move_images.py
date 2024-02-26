from os.path import basename, join
from utils.util import get_images
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    root = r'D:\data\0danyang\20240113\1.13-C2\good'
    target = r'D:\data\0danyang\20240113\1.13-C2\0'
    images=get_images(root)
    for image in tqdm(images):
        base_name = basename(image)
        new_path = join(target, base_name)
        shutil.move(image, new_path)
