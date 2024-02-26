# coding=utf-8
'''
@author: zhuanghaizhan
@contact: zhuanghaizhan@seeking.ai
@file: movePicsBytargetID.py
@time: 2021/5/29 6:37
@desc:
'''

import os
import shutil


def get_image_paths(src_dir):
    images = []
    for home, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename.split('.')[-1] in {'jpg', 'jpeg', 'png', 'bmp'}:
                path = os.path.join(home, filename)
                images.append(path)
    return images


def main(src_dir, dst_dir, target_ids):
    paths = get_image_paths(src_dir)
    for path in paths:
        file_name = os.path.split(path)[-1]

        for i in target_ids:
            if 'TARGET' + str(i) + '-' in file_name:
                new_dir = os.path.join(dst_dir, str(i))
                os.makedirs(new_dir, exist_ok=True)
                dst_path = os.path.join(new_dir, file_name)
                # shutil.copy(path,dst_path)
                shutil.move(path, dst_path)

        # print(file_name)


if __name__ == '__main__':
    src_dir = r'F:\dataset\su_SAPA\all_data\20210704\good_up\media\seeking\Data\history\good_up\2021-07-04'
    dst_dir = r'F:\dataset\su_SAPA\all_data\20210704\good-up-split'

    cam_type = 'up'

    if cam_type == 'up':
        # target_ids = {0, 1, 2, 3, 4, 5, 6,15, 24, 33, 42, 51, 60, 69, 78, 87, 96, 105, 114, 123, 124, 125, 126, 127, 128,
        #                 129, 130, 131,13, 14, 23, 32, 41, 50, 59, 68, 77, 86, 95, 104, 113, 122, 131}
        # target_ids = { 25, 34, 43, 52, 61, 70, 79, 88, 97, 106,
        #               17, 26, 35, 44, 53, 62, 71, 80, 89, 98, 107,
        #               18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108,
        #               19, 28, 37, 46, 55, 64, 73, 82, 91, 100, 109,
        #               20, 29, 38, 47, 56, 65, 74, 83, 92, 101, 110,
        #               21, 30, 39, 48, 57, 66, 75, 84, 93, 102, 111,
        #               120,31, 40, 49, 58, 67, 76, 85, 94, 103, 112}
        target_ids = [i for i in range(133)]
    else:
        # target_ids = {0, 1, 2, 3, 4, 5, 6, 7,8,9,18,27,36,45,54,63,72,81,90,99,108,117,
        #               17,26,35,44,53,62,71,80,89,98,107,116,124,125,126,127,128,129}

        # target_ids = {19,20,21,22,23,24,25,28,29,30,31,32,33,34,37,38,39,40,41,42,43,46,47,48,49,50,51,52,55,56,57,58,59,
        # 60,61,64,65,66,67,68,69,70,73,74,75,76,77,78,79,82,83,84,85,86,87,88,91,92,93,94,95,96,97,100,101,102,103,104,105,106,
        # 110,111,112,113,114}
        target_ids = [i for i in range(131)]

    main(src_dir, dst_dir, target_ids)
    print('done')
