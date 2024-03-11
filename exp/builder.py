import os
import random

import torch
import torch.cuda
import torch.backends.cudnn
import numpy as np
from loguru import logger
import torch.optim.lr_scheduler as torch_lr_scheduler

from .model import Model
from .optim import AmpOptimWrapper
from utils.util import get_time
import lr_scheduler.lr_adjustment as lr_adjustment
from .loss_forward import BaseLossRunner, LOSS_FORWARD_TABLE


def build_dir(path: str)->None:
    if os.path.exists(path) is False:
        os.makedirs(path)
    logger.success(f'Create dir:{path}')


def init_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f'Init seed:{seed}.')


def init_backends_cudnn(deterministic: bool = False) -> None:
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(model_args: dict) -> Model:
    model = Model(**model_args)
    model.init_model()
    model.move_to_device()
    return model


def build_optimizer(name: str, **kwargs):
    optim = torch.optim.__dict__.get(name)
    if optim is None:
        logger.error(f'Do not get the {name} optimizer from torch.optim.')
        exit()

    optimizer = optim(**kwargs)

    logger.success(f'Build optimizer: {name} Done.')
    return optimizer


def build_amp_optimizer_wrapper(name: str, **kwargs) -> AmpOptimWrapper:
    optimizer = build_optimizer(name, **kwargs)
    amp_optimizer_wrapper = AmpOptimWrapper(optimizer=optimizer)

    logger.success(f'Build AmpOptimWrapper: {name} Done.')
    return amp_optimizer_wrapper


def build_lr_scheduler(name: str, **kwargs):
    lr_scheduler = lr_adjustment.__dict__.get(name)

    if lr_scheduler is None:
        lr_scheduler = torch_lr_scheduler.__dict__.get(name)

    elif lr_scheduler is None:
        logger.error(f'Do not get the {name} lr_scheduler.')
        exit()

    lr_scheduler = lr_scheduler(**kwargs)
    logger.success(f'Build lr scheduler: {name} Done.')
    return lr_scheduler


def build_loss(name, **kwargs) -> BaseLossRunner:
    loss_forward = LOSS_FORWARD_TABLE.get(name)
    loss_forward_obj = loss_forward(**kwargs)
    loss_forward_obj.build()
    return loss_forward_obj
