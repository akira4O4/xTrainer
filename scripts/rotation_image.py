from utils.util import get_images, check_dir
import cv2
import os
from tqdm import tqdm
import shutil

if __name__ == '__main__':
    rotation_angle = {
        '90': cv2.ROTATE_90_CLOCKWISE,
        '180': cv2.ROTATE_180,
        '270': cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    need_angle = ['90', '180', '270']
    img_dir = r'D:\llf\dataset\gz_sapa_0\gz_sapa_jiayaosai_duanmian\classification\train\1_posun\20230327\input'
    output_dir = r'D:\llf\dataset\gz_sapa_0\gz_sapa_jiayaosai_duanmian\classification\train\1_posun\20230327\rotation'

    check_dir(output_dir)

    images = get_images(img_dir)
    for image in tqdm(images):
        im = cv2.imread(image)
        for angle in need_angle:
            ret = cv2.rotate(im, rotateCode=rotation_angle[angle])

            image_basename = os.path.basename(image)
            name, ext = os.path.splitext(image_basename)
            image_output_name = name + f'_r{angle}' + ext
            image_save_path = os.path.join(output_dir, image_output_name)

            cv2.imwrite(image_save_path, ret)
