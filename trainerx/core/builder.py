import torch
import torch.cuda
import torch.backends.cudnn
from torch.optim import Optimizer

from trainerx.core.optim import OptimWrapper, AMPOptimWrapper
from trainerx.core import loss_forward


# from trainerx.core.loss_forward import LOSS_FORWARD_TABLE


def build_optimizer(name: str, **kwargs) -> Optimizer:
    optim = torch.optim.__dict__.get(name)
    assert optim is not None
    optimizer = optim(**kwargs)
    return optimizer


def build_optimizer_wrapper(name: str, **kwargs) -> OptimWrapper:
    optimizer = build_optimizer(name, **kwargs)
    optimizer_wrapper = OptimWrapper(optimizer=optimizer)
    return optimizer_wrapper


def build_amp_optimizer_wrapper(name: str, **kwargs) -> AMPOptimWrapper:
    optimizer = build_optimizer(name, **kwargs)
    amp_optimizer_wrapper = AMPOptimWrapper(optimizer=optimizer)
    return amp_optimizer_wrapper


def build_loss_forward(name: str, **kwargs):
    loss_forward_class = loss_forward.__dict__.get(name)
    assert loss_forward_class is not None

    loss_forward_instance = loss_forward_class(**kwargs)
    loss_forward_instance.build()
    return loss_forward_instance
