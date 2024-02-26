import os
from random import sample
from utils.util import get_images
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    root = r'D:\llf\dataset\gz_sapa_0\suzhou_sapa_jiayaosai_duanmian\classification\train\0_good\suzou'
    output = r'D:\llf\dataset\gz_sapa_0\suzhou_sapa_jiayaosai_duanmian\classification\not_used\0_good'
    random_rate = 0.1

    images = get_images(root)
    imaegs_len = len(images)
    new_data = sample(images, int(imaegs_len * random_rate))

    for image in tqdm(new_data):
        image_basename = os.path.basename(image)
        new_image_path = image.replace(root, output)
        new_image_dir = new_image_path.split(image_basename)[0]
        if not os.path.exists(new_image_dir):
            os.makedirs(new_image_dir, exist_ok=True)
        shutil.move(image, new_image_dir)
