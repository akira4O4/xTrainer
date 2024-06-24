import os
import random
from typing import Optional
import torch
import torch.cuda
import torch.backends.cudnn
import numpy as np
# from loguru import logger
import torch.optim.lr_scheduler as torch_lr_scheduler
from torch.optim import Optimizer
from .model import Model
from .optim import OptimWrapper, AmpOptimWrapper
from src import lr_scheduler as lr_adjustment
from .loss_forward import BaseLossForward, LOSS_FORWARD_TABLE
import warnings


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


def build_model(model_args: dict) -> Model:
    warnings.warn("", DeprecationWarning)
    model = Model(**model_args)
    model.init_model()
    model.move_to_device()
    return model


# kwargs=official args
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


def build_loss_forward(name, **kwargs) -> BaseLossForward:
    loss_forward_class = LOSS_FORWARD_TABLE.get(name)
    assert loss_forward_class is not None

    loss_forward_instance = loss_forward_class(**kwargs)
    loss_forward_instance.build()
    return loss_forward_instance
