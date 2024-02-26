import io
import base64
import json
import shutil, os, glob
import time
from datetime import datetime
import cv2
import yaml
import torch
import numpy as np

from loguru import logger
from PIL import Image


# 计算混淆矩阵
def generate_matrix(classes, output, target):
    with torch.no_grad():
        target = target.squeeze(1).cpu().detach().numpy()
        pred = output.argmax(1).cpu().detach().numpy()
        mask = (target >= 0) & (target < classes)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        label = classes * target[mask].astype('int') + pred[mask]
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=classes ** 2)
        confusion_matrix = count.reshape(classes, classes)  # (n, n)
    return confusion_matrix


def accl_miou(hist):
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)
    return iou, miou


def save_class_id_map(class_id_map_save_path: str, classes2idx: dict):
    with open(class_id_map_save_path, 'w') as f:
        f.writelines(['{}x{}\n'.format(v, k) for k, v in classes2idx.items()])


def load_checkpoint(weight: str):
    checkpoint = torch.load(weight)
    return checkpoint


def load_weight(model, weight: str, strict: bool = False):
    """
    just load model state_dict
    """
    if os.path.exists(weight):
        checkpoint = torch.load(weight)
        static_dict = checkpoint.get('state_dict')
        model.load_state_dict(static_dict, strict=strict)
        logger.success(f"{weight} 权重resume成功。")
    else:
        logger.warning(f"{weight} 权重文件不存在。")


def load_loss(loss, weight, strict: False):
    checkpoint = torch.load(weight)
    center_loss_para = checkpoint['center_loss']
    loss.load_state_dict(center_loss_para, strict=strict)
    logger.success(f"loss update.")


def save_weight(state_dict: dict, path: str):
    cur_time = get_time()
    filename = os.path.join(path, cur_time + '_checkpoint.pth')
    if not os.path.exists(path):
        os.mkdir(path)
        logger.info("create path {path} to save checkpoint.")
    torch.save(state_dict, filename)


def get_time(fmt: str = '%Y-%m-%d-(%H:%M:%S)') -> str:
    time_str = time.strftime(fmt, time.localtime())
    # return '2022 - 11 - 14 - (17:35:42)'
    return str(time_str)


def auto_mkdir_project(project_name: str):
    sub_dir = ['export', 'runs', 'weights', 'log']
    dirs = [os.path.join('project', project_name, item) for item in sub_dir]
    for d in dirs:
        if os.path.exists(d):
            continue
        os.makedirs(d)
    logger.info(f'Create {project_name}')


def get_labelme_ann(image, seg, labelmap, cls_):
    json_data = {}
    shapes = []
    version = "4.5.9"
    flags = {}
    imagePath = image.split(os.sep)[-1]
    img_arr = cv2.imread(image)[:, :, ::-1]
    imageData = img_tobyte(Image.fromarray(img_arr))
    h, w = img_arr.shape[0], img_arr.shape[1]

    for i in range(1, cls_):
        sub_mask = np.zeros((h, w), dtype="uint8")
        sub_mask[seg == i] = 1
        contour, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            if len(cnt) < 3:
                continue
            seg_info = {}
            seg_info["label"] = str(i) + labelmap[i]
            seg_info["points"] = cnt.reshape(-1, 2).tolist()
            # print(seg_info["points"].shape)
            seg_info["group_id"] = None
            seg_info["shape_type"] = "polygon"
            seg_info["flags"] = {}
            shapes.append(seg_info)

    json_data["shapes"] = shapes
    json_data["version"] = version
    json_data["flags"] = flags
    json_data["imagePath"] = imagePath
    json_data["imageData"] = imageData
    json_data["imageHeight"] = h
    json_data["imageWidth"] = w

    return json_data


def get_images(path, ext=None):
    if ext is None:
        ext = ["jpg", "png", "jpeg"]

    ret = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_extend = file.split(".")[-1]
            if file_extend in ext:
                ret.append(os.path.join(root, file))
    return ret


def distance(embeddings1, embeddings2):
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    return dist


def img_tobyte(img_pil):
    ENCODING = 'utf-8'
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format='PNG')
    binary_str2 = img_byte.getvalue()
    imageData = base64.b64encode(binary_str2)
    base64_string = imageData.decode(ENCODING)
    return base64_string


def write_json(json_data, save_path):
    with open(save_path, "w") as f:
        json.dump(json_data, f)


def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
    return json_data


def display_config(data):
    for k, v in data.items():
        if k == 'augment':
            continue
        logger.info('{:30} : {}'.format(k, v))

    logger.info('')

    for k, v in data['augment'].items():
        logger.info('{:30} : {}'.format(k, v))


def load_aug_config(params) -> list:
    use_aug_params = [k for k, v in params.items() if v]
    return use_aug_params


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        # output.size()=(bs,nc)
        # target.size()=(bs,nc)
        batch_size = target.size(0)

        # max_val,idx=touch.topk()
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # (bs,nc)->(nc,bs)
        # target:(bs,nc)->(1,bsxnc)->(nc,bs)
        #
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_balance_weight(beta, samples_per_cls, classes):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * classes
    weights = torch.tensor(weights).float()
    return weights


def vis_maps(img, predict, num_of_class, save_path):
    color_list = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                  [255, 0, 85], [255, 0, 170],
                  [0, 255, 0], [85, 255, 0], [170, 255, 0],
                  [0, 255, 85], [0, 255, 170],
                  [0, 0, 255], [85, 0, 255], [170, 0, 255],
                  [0, 85, 255], [0, 170, 255],
                  [255, 255, 0], [255, 255, 85], [255, 255, 170],
                  [255, 0, 255], [255, 85, 255], [255, 170, 255],
                  [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    img = np.array(img)
    vis_img = img.copy().astype(np.uint8)
    vis_predict = predict.copy().astype(np.uint8)
    vis_predict = cv2.resize(vis_predict, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    vis_predict_color = np.zeros((img.shape[0], img.shape[1], 3)) + 255
    for pi in range(0, num_of_class):
        index = np.where(vis_predict == pi)
        vis_predict_color[index[0], index[1], :] = color_list[pi]

    vis_predict_color = vis_predict_color.astype(np.uint8)
    vis_opencv = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    vis_addweight = cv2.addWeighted(vis_opencv, 0.4, vis_predict_color, 0.6, 0)
    cv2.imwrite(save_path[:-4] + ".png", vis_opencv)
    cv2.imwrite(save_path[:-4] + ".jpg", vis_addweight)


# move to evaluation.py
def accuracy_good_ng(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred_copy = pred.clone()
        pred_copy[pred_copy > 0] = 1
        target_copy = target.clone()
        target_copy[target_copy > 0] = 1
        correct = pred_copy.eq(target_copy.view(1, -1).expand_as(pred_copy))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, train_network_cls, filename='temp/checkpoint'):
    best_filename = "temp/model_best"
    if train_network_cls:
        filename = filename + "_landmark_.pth.tar"
        best_filename = best_filename + "_landmark_.pth.tar"
    else:
        filename = filename + "_cls_.pth.tar"
        best_filename = best_filename + "_cls_.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def make_dir(root_path, clear_flag=True):
    if os.path.exists(root_path):
        if clear_flag:
            shutil.rmtree(root_path)
            os.makedirs(root_path)
    else:
        os.makedirs(root_path)


def get_key(dct, value):
    return list(filter(lambda k: dct[k] == value, dct))


# move to evaluation.py
# accuracymultilabel
def accuracymultilabel(output, target):
    with torch.no_grad():
        num_instance, num_class = target.size()
        output[output > 0.5] = 1
        output[output <= 0.5] = 0

        count = 0
        gbcount = 0
        for i in range(num_instance):
            p = sum(np.logical_and(target[i].cpu().detach().numpy(), output[i].cpu().detach().numpy()))
            q = sum(np.logical_or(target[i].cpu().detach().numpy(), output[i].cpu().detach().numpy()))
            count += p / q

            if output[i][0] == target[i][0]:
                gbcount += 1

    return [gbcount / num_instance], [count / num_instance]


def load_yaml(yaml_path: str) -> dict:
    """
    :param yaml_path: xxx/xxx/xxx.yaml
    """
    assert yaml_path.endswith(".yaml") is True, f'file type must be yaml'
    assert os.path.exists(yaml_path) is True, f'{yaml_path}is error'
    with open(yaml_path, 'r', encoding='UTF-8') as f:
        conf = yaml.safe_load(f)
    return conf


if __name__ == '__main__':
    a = get_time()
    print(a)
