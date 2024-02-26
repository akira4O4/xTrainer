import os
import random
from utils.util import get_images
import shutil
from tqdm import tqdm


def subset(alist: list, idxs: list) -> list:
    # 用法：根据下标idxs取出列表alist的子集
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list


def split_list(
        alist: list,
        group_num: int = 4,
        shuffle: bool = True,
        retain_left: bool = True
) -> dict:
    """
        用法：将alist切分成group个子列表，每个子列表里面有len(alist)//group个元素
        shuffle: 表示是否要随机切分列表，默认为True
        retain_left: 若将列表alist分成group_num个子列表后还要剩余，是否将剩余的元素单独作为一组
    """

    index = list(range(len(alist)))  # 保留下标

    # 是否打乱列表
    if shuffle:
        random.shuffle(index)

    elem_num = len(alist) // group_num  # 每一个子列表所含有的元素数量
    sub_lists = {}

    # 取出每一个子列表所包含的元素，存入字典中
    for idx in range(group_num):
        start, end = idx * elem_num, (idx + 1) * elem_num
        sub_lists['set' + str(idx)] = subset(alist, index[start:end])

    # 是否将最后剩余的元素作为单独的一组
    if retain_left and group_num * elem_num != len(index):  # 列表元素数量未能整除子列表数，需要将最后那一部分元素单独作为新的列表
        sub_lists['set' + str(idx + 1)] = subset(alist, index[end:])

    return sub_lists


if __name__ == '__main__':
    root = r'D:\data\danyang\20240111\20240111-B\good-f'
    output = r'D:\data\danyang\20240111\20240111-B\good-f-split'
    group_num = 3

    images = get_images(root)
    split_data = split_list(images, group_num)

    for k, item in split_data.items():
        for image in tqdm(item):

            new_dir = os.path.join(output, k)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            basename = os.path.basename(image)
            new_path = os.path.join(new_dir, basename)
            shutil.copy(image, new_path)
