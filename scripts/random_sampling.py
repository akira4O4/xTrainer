import os
from random import sample
from utils.util import get_images
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    root = r'D:\llf\dataset\dog_cat\all\train\0cat'
    output = r'D:\llf\dataset\dog_cat\1280\train\0cat'
    # root = r'D:\llf\dataset\danyang\F_train\seg\train\3_pobian\1123\3_pobian'
    # output = r'D:\llf\dataset\danyang\F_train\seg\128\train'
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    random_rate = 1280
    remove_json = True

    images = get_images(root)
    images_len = len(images)

    if random_rate > 1:
        k = random_rate
    else:  # k%
        k = int(images_len * random_rate)
    new_data = sample(images, k)

    for image in tqdm(new_data):
        image_basename = os.path.basename(image)
        new_image_path = image.replace(root, output)
        # new_image_dir = new_image_path.split(image_basename)[0]
        # if not os.path.exists(new_image_dir):
        #     os.makedirs(new_image_dir, exist_ok=True)

        shutil.copy(image, new_image_path)

        if remove_json:
            name, ext = os.path.splitext(image_basename)
            json_file = image.replace(ext, '.json')
            if not os.path.exists(json_file):
                continue

            new_json_path = json_file.replace(root, output)
            shutil.copy(json_file, new_json_path)
