import os
import os.path as osp
import time
import json
import shutil
import glob
from typing import Union, Optional
from bunch import Bunch

import cv2
import torch
import numpy as np
from loguru import logger
import yaml
from loguru import logger
from enum import Enum, unique
import warnings
__all__ = [
    'split_line',
    'letterbox', 'check_exists', 'check_dir', 'recursion_bunch', 'check_ext',
    'load_json', 'load_yaml', 'load_config', 'get_time', 'get_file_list',
    'accuracy', 'get_balance_weight', 'generate_matrix', 'accl_miou',
    'get_num_of_images', 'get_images', 'get_file_with_ext', 'get_file_with_ext',
    'timer', 'Task', 'join', 'task_convert'
]
color_list = [
    [0, 0, 0],
    [0, 255, 0], [0, 0, 255], [0, 255, 255],
    [255, 255, 0], [0, 255, 255], [255, 255, 0],
    [255, 255, 255], [170, 255, 255],
    [255, 0, 170], [85, 0, 255], [128, 255, 128],
    [170, 255, 255], [0, 255, 170], [85, 0, 255],
    [170, 0, 255], [0, 85, 255], [0, 170, 255],
    [255, 255, 85], [255, 255, 170], [255, 0, 255],
    [255, 85, 255], [255, 170, 255], [85, 255, 255],
]


def round4(data):
    return round(float(data), 4)


def round8(data):
    return round(float(data), 8)


@unique
class Task(Enum):
    CLS = 'classification'
    SEG = 'segmentation'
    MultiTask = 'multitask'


def join(*args, **kwargs) -> str:
    return osp.join(*args, **kwargs)


def task_convert(task: str) -> Task:
    if task == 'multitask':
        return Task.MultiTask
    elif task == 'classification':
        return Task.CLS
    elif task == 'segmentation':
        return Task.SEG


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print(f'🕐 {func.__name__} Spend Time: {format(time_spend, ".3f")}s')
        return result

    return func_wrapper





def split_line(num: int = 50):
    logger.debug('+' + '-' * num + '+')


def letterbox(
    image_src: np.ndarray,
    dst_size: Union[tuple, list],
    pad_color: Optional[Union[tuple, list]] = (114, 114, 114)
) -> tuple:
    src_h, src_w = image_src.shape[:2]
    dst_h, dst_w = dst_size
    scale = min(dst_h / src_h, dst_w / src_w)
    pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))

    if image_src.shape[0:2] != (pad_w, pad_h):
        image_dst = cv2.resize(image_src, (pad_w, pad_h),
                               interpolation=cv2.INTER_LINEAR)
    else:
        image_dst = image_src

    top = int((dst_h - pad_h) / 2)
    down = int((dst_h - pad_h + 1) / 2)
    left = int((dst_w - pad_w) / 2)
    right = int((dst_w - pad_w + 1) / 2)

    # add border
    image_dst = cv2.copyMakeBorder(
        image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
    return image_dst, x_offset, y_offset


def check_exists(path) -> bool:
    if not os.path.exists(path):
        logger.info(f'{path} is not found.')
        return False
    return True


def check_dir(path: str, clean: bool = False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if clean:
            shutil.rmtree(path)
            os.makedirs(path)


def recursion_bunch(kv: dict) -> Bunch:
    for k, v in kv.items():
        if isinstance(v, dict):
            kv[k] = recursion_bunch(v)
    return Bunch(kv)


def check_ext(file: str, exts: list) -> bool:
    base_name = os.path.basename(file)
    _, ext = os.path.splitext(base_name)
    if ext not in exts:
        logger.error(f'{file} ext is not validate:f{exts}')
        return False
    return True


def load_json(path: str):
    data = None
    if not check_ext(path, ['.json']):
        return data
    with open(path, 'r') as config_file:
        data = json.load(config_file)  # 配置字典
    return data


def load_yaml(path: str):
    data = None
    if not check_ext(path, ['.yaml', '.yml']):
        return data

    with open(path, encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def load_config(config_file: str):
    config_data = load_yaml(config_file)
    config = recursion_bunch(config_data)
    return config


# def vis_maps_v2(image: str,
#                 predict,
#                 classes: int,
#                 wh: tuple,
#                 save_path: str,
#                 sum_method: bool = False,
#                 threshold: list = None,
#                 ignoreindex=None):
#     img = cv2.imread(image)
#     img = cv2.resize(img, wh, interpolation=cv2.INTER_NEAREST)
#     img = img.copy().astype(np.uint8)
#     vis_predict = predict.copy().astype(np.uint8)  # 0-255
#     vis_predict = cv2.resize(vis_predict, None, fx=1,
#                              fy=1, interpolation=cv2.INTER_NEAREST)
#     # shape(w,h,3) color is (255,255,255)
#     vis_predict_seg = np.zeros((img.shape[0], img.shape[1], 3)) + [255, 0, 0]
#
#     ignoreflag = True if ignoreindex is None else False
#     # e.g.{id1:area_size,id2:area_size}
#     seg_classes = {}
#     for pi in range(0, classes):
#         seg_cls_item = []
#         index = np.where(vis_predict == pi)
#         if len(index[0]) > 0 and pi != 0:
#             # print(pi)
#             #     seg_classes.append(pi)
#             ...
#
#         if not ignoreflag:
#             if len(index[0]) > 0 and pi not in ignoreindex:
#                 ignoreflag = True
#
#         if not threshold is None:
#             if not sum_method:  # 單獨輪廓面積過濾
#                 mask_index = np.zeros(
#                     (img.shape[0], img.shape[1]), dtype=np.uint8)
#                 mask_index[index[0], index[1]] = 255
#                 cnts, _ = cv2.findContours(
#                     mask_index, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 draw_cns = []
#                 for cnt in cnts:
#                     area = cv2.contourArea(cnt)
#                     if area >= threshold[pi]:
#                         draw_cns.append(cnt)
#
#                 cv2.drawContours(vis_predict_seg, draw_cns, -
#                 1, tuple(color_list[pi]), -1)
#             else:  # 面積求和面積過濾
#                 if len(index[0]) >= threshold[pi]:
#                     vis_predict_seg[index[0], index[1], :] = color_list[pi]
#         else:
#             vis_predict_seg[index[0], index[1], :] = color_list[pi]
#
#     if not ignoreflag:
#         return None
#
#     vis_predict_color = vis_predict_seg.astype(np.uint8)
#     vis_addweight = cv2.addWeighted(img, 0.6, vis_predict_color, 0.4, 0)
#
#     image_basename = os.path.basename(image)
#     file_name, ext = os.path.splitext(image_basename)
#
#     cv2.imwrite(os.path.join(save_path, image_basename), img)
#     cv2.imwrite(os.path.join(save_path, file_name + '_seg.png'), vis_addweight)
#
#     return vis_predict


def get_time(fmt: str = '%Y%m%d_%H%M%S') -> str:
    time_str = time.strftime(fmt, time.localtime())
    return str(time_str)


def get_file_list(folder_path: str, p_postfix: list = None, sub_dir: bool = True) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :param sub_dir: 是否搜索子文件夹
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/**/*.*', recursive=True) if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return file_list


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple = (1,)
) -> list:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_balance_weight(beta, samples_per_cls, classes: int):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * classes

    weights = torch.tensor(weights).float()
    return weights


# def vis_maps(img,
#              predict,
#              num_of_class: int,
#              save_path: str,
#              sum_method: bool = False,
#              threshold: list = None,
#              ignoreindex: bool = None):
#     color_list = [[255, 0, 0], [255, 255, 255], [255, 0, 170],
#                   [0, 255, 0], [0, 255, 255], [85, 0, 255],
#                   [128, 255, 128], [170, 255, 255], [255, 255, 0],
#                   [0, 255, 170], [0, 0, 255], [85, 0, 255],
#                   [170, 0, 255], [0, 85, 255], [0, 170, 255],
#                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
#                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
#                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
#     img = np.array(img)
#     vis_img = img.copy().astype(np.uint8)
#     vis_predict = predict.copy().astype(np.uint8)  # 0-255
#     vis_predict = cv2.resize(vis_predict, None, fx=1,
#                              fy=1, interpolation=cv2.INTER_NEAREST)
#     # shape(w,h,3) color is (255,255,255)
#     vis_predict_color = np.zeros((img.shape[0], img.shape[1], 3)) + 255
#
#     ignoreflag = True if ignoreindex is None else False
#     seg_classes = []
#     for pi in range(0, num_of_class):
#         index = np.where(vis_predict == pi)
#         if len(index[0]) > 0:
#             seg_classes.append(pi)
#
#         if not ignoreflag:
#             if len(index[0]) > 0 and pi not in ignoreindex:
#                 ignoreflag = True
#
#         if not threshold is None:
#             if not sum_method:  # 單獨輪廓面積過濾
#                 mask_index = np.zeros(
#                     (img.shape[0], img.shape[1]), dtype=np.uint8)
#                 mask_index[index[0], index[1]] = 255
#                 cnts, _ = cv2.findContours(
#                     mask_index, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 draw_cns = []
#                 for cnt in cnts:
#                     area = cv2.contourArea(cnt)
#                     if area >= threshold[pi]:
#                         draw_cns.append(cnt)
#
#                 cv2.drawContours(vis_predict_color, draw_cns, -
#                 1, tuple(color_list[pi]), -1)
#             else:  # 面積求和面積過濾
#                 if np.sum(index[0]) >= threshold[pi]:
#                     vis_predict_color[index[0], index[1], :] = color_list[pi]
#         else:
#             vis_predict_color[index[0], index[1], :] = color_list[pi]
#
#     if not ignoreflag:
#         return None
#     vis_predict_color = vis_predict_color.astype(np.uint8)
#     vis_opencv = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
#     vis_addweight = cv2.addWeighted(vis_opencv, 0.4, vis_predict_color, 0.6, 0)
#     # cv2.imwrite(save_path[:-4]+".png",vis_opencv)
#     # cv2.imwrite(save_path[:-4]+".jpg",vis_addweight)
#     cv2.imencode('.png', vis_opencv)[1].tofile(save_path[:-4] + ".png")
#     cv2.imencode('.jpg', vis_addweight)[1].tofile(save_path[:-4] + ".jpg")
#     # cv2.imwrite(save_path[:-4]+"_mask.jpg",vis_predict_color)
#     return vis_predict


# accuracymultilabel
def accuracymultilabel(output, target):
    with torch.no_grad():
        num_instance, num_class = target.size()
        output[output > 0.5] = 1
        output[output <= 0.5] = 0

        count = 0
        gbcount = 0
        for i in range(num_instance):
            p = sum(np.logical_and(target[i].cpu().detach(
            ).numpy(), output[i].cpu().detach().numpy()))
            q = sum(np.logical_or(target[i].cpu().detach(
            ).numpy(), output[i].cpu().detach().numpy()))
            count += p / q

            if output[i][0] == target[i][0]:
                gbcount += 1

    return [gbcount / num_instance], [count / num_instance]


# 计算混淆矩阵
def generate_matrix(
    classes: int,
    output: torch.Tensor,
    target: torch.Tensor
) -> Optional[np.ndarray]:
    if isinstance(output, torch.Tensor) is False:
        logger.error('output type is not torch.Tensor')
        return
    if isinstance(target, torch.Tensor) is False:
        logger.error('target type is not torch.Tensor')
        return
    if output.shape != target.shape:
        logger.error('output.shape!=target.shape.')
        return

    with torch.no_grad():
        target = target.cpu().detach().numpy()  # [number,number,...]
        pred = output.cpu().detach().numpy()  # [number,number,...]
        # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        mask = (target >= 0) & (target < classes)  # [true,false,...]
        label = classes * target[mask].astype('int') + pred[mask]
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=classes ** 2)
        confusion_matrix = count.reshape(classes, classes)  # (n, n)
    return confusion_matrix


def accl_miou(hist) -> tuple:
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)
    return iou, miou


# def accuracy_good_ng(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         pred_copy = pred.clone()
#         pred_copy[pred_copy > 0] = 1
#         target_copy = target.clone()
#         target_copy[target_copy > 0] = 1
#         correct = pred_copy.eq(target_copy.view(1, -1).expand_as(pred_copy))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#

# def save_checkpoint(state, is_best, train_network_cls, filename='temp/checkpoint'):
#     best_filename = "temp/model_best"
#     if train_network_cls:
#         filename = filename + "_landmark_.pth.tar"
#         best_filename = best_filename + "_landmark_.pth.tar"
#     else:
#         filename = filename + "_cls_.pth.tar"
#         best_filename = best_filename + "_cls_.pth.tar"
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, best_filename)


# def make_dir(root_path, clear_flag=True):
#     if os.path.exists(root_path):
#         if clear_flag:
#             shutil.rmtree(root_path)
#             os.mkdir(root_path)
#     else:
#         os.mkdir(root_path)


def get_key(dct, value):
    return list(filter(lambda k: dct[k] == value, dct))


def get_num_of_images(path: str, ext: list = None) -> int:
    images = get_images(path, ext)
    return len(images)


def get_images(path: str, ext=None) -> list:
    if ext is None:
        ext = ['.png', '.jpg']
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext in ext:
                image = os.path.join(root, file)
                data.append(image)
    return data


def get_file_with_ext(path: str, ext: list = None):
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ext is not None:
                file_name, file_ext = os.path.splitext(file)
                if file_ext in ext:
                    image = os.path.join(root, file)
                    data.append(image)
            else:
                image = os.path.join(root, file)
                data.append(image)
    return data

# def letterbox(image_src, dst_size, pad_color=(114, 114, 114)):
#     src_h, src_w = image_src.shape[:2]
#     dst_h, dst_w = dst_size
#     scale = min(dst_h / src_h, dst_w / src_w)
#     pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))
#
#     if image_src.shape[0:2] != (pad_w, pad_h):
#         image_dst = cv2.resize(image_src, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
#     else:
#         image_dst = image_src
#
#     top = int((dst_h - pad_h) / 2)
#     down = int((dst_h - pad_h + 1) / 2)
#     left = int((dst_w - pad_w) / 2)
#     right = int((dst_w - pad_w + 1) / 2)
#
#     # add border
#     image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)
#
#     x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
#     return image_dst, x_offset, y_offset


# if __name__ == '__main__':
#     image_dir = r'/home/llf/桌面/temp'
#     images = get_images(image_dir)
#     for i, image in enumerate(images):
#         image = cv2.imread(image)
#         img, a, b = letterbox(image, (64, 64))
#         cv2.imwrite(os.path.join(image_dir, f'{str(i)}.png'), img)
#         print(a, b)
