import os
import shutil

from utils.util import get_images

if __name__ == '__main__':
    target_dir = r'D:\data\0danyang\20240124\B\bad\E\20240124_seg_已完成'
    output_dir = r'D:\data\0danyang\20240124\nojson'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = get_images(target_dir)
    for image in images:
        basename = os.path.basename(image)
        name, ext = os.path.splitext(basename)
        json_file = image.replace(ext, '.json')
        if not os.path.exists(json_file):
            print(f'Remove image: {image}')
            new_path = os.path.join(output_dir, basename)
            shutil.move(image, new_path)
