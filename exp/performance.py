from typing import Optional
import numpy as np
import torch
from .task import Task
from utils.util import accl_miou, accuracy, generate_matrix


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
        'acc1': -1,
        'accn': -1,
        'miou': -1
    }

    if task == Task.SEG:
        miou = calc_miou(mask_classes, model_output, targets)
        performance['miou'] = miou.item()

    elif task == Task.CLS:
        acc1, accn = calc_accuracy(topk, model_output, targets)

        performance['acc1'] = acc1.item()
        performance['accn'] = accn.item()

    return performance
