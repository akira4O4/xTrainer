import warnings

import torch
import torch.cuda
import torch.backends.cudnn
import torch.optim.lr_scheduler as torch_lr_scheduler
from torch.optim import Optimizer

from trainerx.core.model import Model
from trainerx.core.optim import OptimWrapper, AmpOptimWrapper
from trainerx.core.loss_forward import BaseLossForward, LOSS_FORWARD_TABLE


def build_optimizer(name: str, **kwargs) -> Optimizer:
    optim = torch.optim.__dict__.get(name)
    assert optim is not None
    optimizer = optim(**kwargs)
    return optimizer


def build_amp_optimizer_wrapper(name: str, **kwargs) -> AmpOptimWrapper:
    optimizer = build_optimizer(name, **kwargs)
    amp_optimizer_wrapper = AmpOptimWrapper(optimizer=optimizer)
    return amp_optimizer_wrapper


def build_optimizer_wrapper(name: str, **kwargs) -> OptimWrapper:
    optimizer = build_optimizer(name, **kwargs)
    optimizer_wrapper = OptimWrapper(optimizer=optimizer)
    return optimizer_wrapper


def build_lr_scheduler(name: str, **kwargs):
    lr_scheduler = torch_lr_scheduler.__dict__.get(name)
    assert lr_scheduler is not None
    scheduler = lr_scheduler(**kwargs)
    return scheduler


def build_loss_forward(name: str, **kwargs) -> BaseLossForward:
    loss_forward_class = LOSS_FORWARD_TABLE.get(name)
    assert loss_forward_class is not None

    loss_forward_instance = loss_forward_class(**kwargs)
    loss_forward_instance.build()
    return loss_forward_instance
