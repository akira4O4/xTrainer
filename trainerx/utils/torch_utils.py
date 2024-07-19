import random
import numpy as np
import torch
from torch import nn
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


# def np2torch(data: np.ndarray) -> torch.Tensor:
#     return torch.from_numpy(data)
#
#
# def npimage2torch(img: np.ndarray) -> torch.Tensor:
#     if len(img.shape) < 3:  # image is gray type
#         img = np.expand_dims(img, -1)  # HW->HW1
#
#     # np.ndarray:HWC
#     # torch.Tensor:CHW
#     img = img.transpose(2, 0, 1)
#     img = np.ascontiguousarray(img[::-1])
#     img = torch.from_numpy(img)
#     return img



