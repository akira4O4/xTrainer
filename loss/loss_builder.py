# from torch.nn import CrossEntropyLoss
from loss.crossentropy_loss import CrossEntropyLoss
from loss.iou_ssim_loss_class import IouSsimLoss
from loss.focal_loss_class import FocalLoss
from loss.dice_loss import DiceLoss
from loss.period_loss import PeriodLoss

SUPPORT_LOSS = {"CrossEntropyLoss",
                "FocalLoss",
                "IouSsimLoss",
                "DiceLoss",
                "PeriodLoss"
                }


def build_loss(loss_name, **kwargs):
    assert loss_name in SUPPORT_LOSS, f"all support loss is {SUPPORT_LOSS}"
    criterion = eval(loss_name)(**kwargs)
    return criterion
