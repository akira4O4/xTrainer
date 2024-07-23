from typing import Optional
import numpy as np
import torch
from trainerx.utils.task import Task
from loguru import logger


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


def accl_miou(hist) -> tuple:
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)
    return iou, miou


# Classification accuracy
def calc_accuracy(
    topk: int,
    model_output,
    targets: torch.Tensor
) -> tuple:
    if isinstance(model_output, list):

        model_output = model_output[0]
        if isinstance(model_output, list):
            model_output = model_output[0]

    acc1, acc_n = accuracy(model_output, targets, topk=(1, topk))
    return acc1, acc_n


# Segmentation accuracy(MIoU)
def calc_miou(
    mask_classes: int,
    model_output,
    targets: torch.Tensor
) -> np.ndarray:
    if isinstance(model_output, list):

        model_output = model_output[1]
        if isinstance(model_output, list):
            model_output = model_output[0]

    output = model_output.argmax(1)  # [bs,cls,h,w]->[bs,h,w]
    mask_target_seg = targets.squeeze(1)  # [bs,1,h,w]->[bs,h,w]

    confusion_matrix = generate_matrix(
        mask_classes,
        output,
        mask_target_seg
    )
    iou, miou = accl_miou(confusion_matrix)
    return miou


def calc_performance(
    task: Task,
    topk: Optional[int] = 2,
    mask_classes: Optional[int] = 0,
    model_output=None,
    targets: torch.Tensor = None
) -> dict:
    performance = {
        'top1': -1,
        'topk': -1,
        'miou': -1
    }

    if task == Task.SEG:
        miou = calc_miou(mask_classes, model_output, targets)
        performance['miou'] = miou.item() * 100

    elif task == Task.CLS:
        acc1, accn = calc_accuracy(topk, model_output, targets)

        performance['top1'] = acc1.item()
        performance['topk'] = accn.item()

    return performance
