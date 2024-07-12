import random
import numpy as np
import torch
import torch.backends.cudnn


def init_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def init_backends_cudnn(deterministic: bool = False) -> None:
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def iou(pred: torch.Tensor, target: torch.Tensor):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return self._iou(pred, target)

    def _iou(self, pred: torch.Tensor, target: torch.Tensor):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1

            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)

        return IoU / b


def np2torch(data: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(data)


def npimage2torch(img: np.ndarray) -> torch.Tensor:
    if len(img.shape) < 3:  # image is gray type
        img = np.expand_dims(img, -1)  # HW->HW1

    # np.ndarray:HWC
    # torch.Tensor:CHW
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img[::-1])
    img = torch.from_numpy(img)
    return img
