import os
import random

import torch
import torch.cuda
import torch.backends.cudnn
import numpy as np
from loguru import logger
import torch.optim.lr_scheduler as torch_lr_scheduler
from torch.utils.data import DataLoader

from task import Task
from model import Model
from optim import AmpOptimWrapper
import lr_scheduler.lr_adjustment as lr_adjustment
from dataset import ClassificationDataset, SegmentationDataSet


def build_workspace(work_dir: str) -> None:
    if os.path.exists(work_dir) is False:
        logger.info(f'Create work dir:{work_dir}')
        os.makedirs(work_dir)

    logger.info('Init workspace done.')


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

    logger.info(f'Build optimizer: {name}')
    return optimizer


def build_amp_optimizer_wrapper(name: str, **kwargs) -> AmpOptimWrapper:
    optimizer = build_optimizer(name, **kwargs)
    amp_optimizer_wrapper = AmpOptimWrapper(optimizer=optimizer)

    logger.info(f'Build AmpOptimWrapper: {name}')
    return amp_optimizer_wrapper


def build_lr_scheduler(name: str, **kwargs):
    lr_scheduler = lr_adjustment.__dict__.get(name)

    if lr_scheduler is None:
        lr_scheduler = torch_lr_scheduler.__dict__.get(name)

    elif lr_scheduler is None:
        logger.error(f'Do not get the {name} lr_scheduler.')
        exit()

    lr_scheduler = lr_scheduler(**kwargs)
    logger.info(f'Build lr scheduler: {name}')
    return lr_scheduler


def build_classification_dataloader(dataset_args: dict, dataloader_args: dict) -> DataLoader:
    dataset = ClassificationDataset(**dataset_args)
    dataloader = DataLoader(dataset=dataset, **dataloader_args)
    return dataloader


def build_segmentation_dataloader(dataset_args: dict, dataloader_args: dict) -> DataLoader:
    dataset = SegmentationDataSet(**dataset_args)
    dataloader = DataLoader(dataset=dataset, **dataloader_args)
    return dataloader


def build_training_logger():
    ...
