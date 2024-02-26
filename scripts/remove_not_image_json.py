import os
import shutil
from tqdm import tqdm
from utils.util import get_images, get_file_with_ext
import os

if __name__ == '__main__':
    target_dir = r'C:\Users\Administrator\Desktop\tmp\w'
    output_dir = r'C:\Users\Administrator\Desktop\tmp\json'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = get_file_with_ext(target_dir, ['.json'])
    cnt = 0
    for json_file in tqdm(json_files):
        image_file = json_file.replace('json', 'jpg')
        basename = os.path.basename(json_file)

        if os.path.exists(image_file) is False:
            shutil.move(json_file, os.path.join(output_dir, basename))
            cnt += 1
    print(f'Remove json: {cnt}')
